import tensorflow as tf
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import scipy.stats
import sys
import os
from dsn_util import time_invariant_flow, check_convergence, setup_param_logging, \
                      initialize_optimization_parameters, computeMoments, getEtas, \
                      approxKL, setup_IO, compute_R2, AR_to_autocov_np, \
                      AL_cost, log_grads, memory_extension 
from tf_util.tf_util import construct_flow, declare_theta, connect_flow, count_params
import datetime
import io
from sklearn.metrics import pairwise_distances
import pandas as pd
import seaborn as sns

def train_dsn(system, behavior, n, flow_dict, k_max=10, c_init=1e0, lr_order=-3, check_rate=100, max_iters=5000, random_seed=0):
    D = system.D;

    opt_method = 'adam';
    dynamics = None;
    num_zi = 1;

    # set initialization of AL parameter c and learning rate
    lr = 10**lr_order
    print('c_init', c_init);

    # good practice
    tf.reset_default_graph();
    tf.set_random_seed(random_seed);

    # save tensorboard summary in intervals
    tb_save_every = 1;
    model_save_every = 50000;

    # AL switch criteria
    stop_early = True;

    min_iters = 1000;
    cost_grad_lag = 100;
    pthresh = 0.05;
    #sigma_eps_buf = 1e-8;
    tb_save_params = True;

    savedir = setup_IO(system, flow_dict, random_seed);
    print(savedir);


    flow_layers, Z0, Z_AR, base_log_q_x, num_theta_params = \
        construct_flow(flow_dict, D, 1);
    flow_layers, num_theta_params = system.map_to_parameter_support(flow_layers, num_theta_params);
    Z0_shape = tf.shape(Z0);
    batch_size = tf.multiply(Z0_shape[0], Z0_shape[1]);

    if (dynamics is not None):
        A, sigma_eps = dyn_pars;
    
    theta = declare_theta(flow_layers);
    # connect time-invariant flow
    phi, sum_log_det_jacobian, Z_by_layer = connect_flow(Z_AR, flow_layers, theta);
    log_q_x = base_log_q_x - sum_log_det_jacobian;

    # generative model is fully specified
    all_params = tf.trainable_variables();
    nparams = len(all_params);

    # set up the constraint computation
    T_x = system.compute_suff_stats(phi);
    mu = system.compute_mu(behavior);
    T_x_mu_centered = system.center_suff_stats_by_mu(T_x, mu);

    if (system.name == 'damped_harmonic_oscillator'):
        XY = system.simulate(phi);
        X = tf.clip_by_value(XY[:,:,0,:], 1e-3, 1e3);

    #R2 = compute_R2(log_q_x, None, T_x_mu_centered);

    # augmented lagrangian optimization
    Lambda = tf.placeholder(dtype=tf.float64, shape=(system.num_suff_stats,));
    c = tf.placeholder(dtype=tf.float64, shape=());
    print('train network');

    cost, cost_grads, H = AL_cost(log_q_x, T_x_mu_centered, Lambda, c, all_params);

    grads_and_vars = [];
    for i in range(len(all_params)):
        grads_and_vars.append((cost_grads[i], all_params[i]));

    # set optimization hyperparameters
    n_tilde = 300;
    saver = tf.train.Saver();
    tf.add_to_collection('Z0', Z0);
    tf.add_to_collection('phi', phi);

    # tensorboard logging
    summary_writer = tf.summary.FileWriter(savedir);
    tf.summary.scalar('H', -tf.reduce_mean(log_q_x));
    tf.summary.scalar('cost', cost);

    # log parameter values throughout optimization
    if (tb_save_params):
        setup_param_logging(all_params);

    nparam_vals = count_params(all_params);

    summary_op = tf.summary.merge_all();

    opt_compress_fac = 16;
    array_init_len = int(np.ceil(max_iters/opt_compress_fac));
    cost_grad_vals = np.zeros((array_init_len, nparam_vals));
    array_cur_len = array_init_len;

    num_diagnostic_checks = k_max*(max_iters // check_rate);
    costs = np.zeros((num_diagnostic_checks,));
    Hs = np.zeros((num_diagnostic_checks,));
    T_x_mu_centereds = np.zeros((num_diagnostic_checks, system.num_suff_stats));
    check_it = 0;
    _c = c_init;
    _lambda = np.zeros((system.num_suff_stats,));
    with tf.Session() as sess:
        print('training DSN for %s: dt=%.3f, T=%d' % (system.name, system.dt, system.T));
        init_op = tf.global_variables_initializer();
        sess.run(init_op);

        z_i = np.random.normal(np.zeros((1,n,D,num_zi)), 1.0);
        feed_dict_R2 = {Z0:z_i, Lambda:_lambda, c:_c};
        _T_x_mu_centered, _phi, _log_q_x = sess.run([T_x_mu_centered, phi, log_q_x], feed_dict_R2);

        """
        print('phi');
        print(_phi);
        print('T(x)');
        print(_T_x_mu_centered.shape);
        print('log_q_x');
        print(_log_q_x);
        """
        z_i = np.random.normal(np.zeros((1,n,D,num_zi)), 1.0);
        feed_dict = {Z0:z_i, Lambda:_lambda, c:_c};

        cost_i, _cost_grads, summary = \
                    sess.run([cost, cost_grads, summary_op], feed_dict);

        _T_x_mu_centered = sess.run(T_x_mu_centered, feed_dict);
        T_x_mu_centereds[0,:] = np.mean(_T_x_mu_centered[0], 0);
        
        log_grads(_cost_grads, cost_grad_vals, 0);
       
        total_its = 1;
        for k in range(k_max):
            print("AL iteration %d" % (k+1));
            # reset optimizer
            print('resetting optimizer');
            if (opt_method == 'adam'):
                optimizer = tf.contrib.optimizer_v2.AdamOptimizer(learning_rate=lr);
            elif (opt_method == 'adadelta'):
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr);
            elif (opt_method == 'adagrad'):
                optimizer = tf.train.AdagradOptimizer(learning_rate=lr);
            elif (opt_method == 'graddesc'):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr);
            else:
                print('Error: No optimization match');

            train_step = optimizer.apply_gradients(grads_and_vars);
            initialize_optimization_parameters(sess, optimizer, all_params);
            slot_names = optimizer.get_slot_names();
            debug_opt_var = optimizer.get_slot(all_params[0], slot_names[0]);

            # SGD iteration
            i = 0;
            has_converged = False;
            #while ((i < (cost_grad_lag)) or (not has_converged)): 
            convergence_it = 0;
            print('aug lag it', k);
            print('lambda', _lambda);
            plt.plot(_lambda);
            print('c', _c);

            while (i < max_iters):
                cur_ind = total_its + i;

                if (cur_ind == array_cur_len):
                    _, _, cost_grad_vals = memory_extension(None, None, cost_grad_vals, array_cur_len);
                    array_cur_len = 2*array_cur_len;

                z_i = np.random.normal(np.zeros((1,n,D,num_zi)), 1.0);
                feed_dict = {Z0:z_i, Lambda:_lambda, c:_c};

                _T_x = sess.run(T_x, feed_dict);

                ts, cost_i, _cost_grads, summary, _T_x, _H = \
                    sess.run([train_step, cost, cost_grads, summary_op, T_x, H], feed_dict);

                _phi, _T_x, _H = sess.run([phi, T_x, H], feed_dict);

                log_grads(_cost_grads, cost_grad_vals, cur_ind);

                if (np.mod(cur_ind,tb_save_every)==0):
                    summary_writer.add_summary(summary, cur_ind);

                if (np.mod(i,model_save_every) == 0):
                    # save all the hyperparams
                    if not os.path.exists(savedir):
                        print('Making directory %s' % savedir );
                        os.makedirs(savedir);
                    print('saving model at iter', i);
                    saver.save(sess, savedir + 'model');

                #if (i > (cost_grad_lag) and np.mod(cur_ind, check_rate)==0):
                if (np.mod(cur_ind, check_rate)==0):
                    _H, _T_x, _T_x_mu_centered, _phi, _log_q_x, = sess.run([H, T_x, T_x_mu_centered, phi, log_q_x], feed_dict);
                    print(42*'*');
                    print('it = %d ' % cur_ind);
                    print('H', _H);
                    mean_T = np.mean(_T_x_mu_centered, 1);
                    #if (system.num_suff_stats > 0):
                        #print('E[T_x - mu] = ', mean_T[0,0], mean_T[0,1]);
                    print('cost', cost_i);

                    if (system.name in ['null_on_interval', 'one_con_on_interval', 'two_con_on_interval']):
                        fig = plt.figure();
                        print('system.D', system.D);
                        for plot_ind in range(system.D):
                            plt.subplot(1,system.D,plot_ind+1);
                            plt.hist(_phi[0,:,plot_ind,0]);
                            plt.title(r'$\phi_%d$' % (plot_ind+1));
                        plt.show();

                        if (system.D == 3):
                            fontsize = 14;
                            k_i = _phi[0,:,0,0];
                            m_i = _phi[0,:,1,0];
                            c_i = _phi[0,:,2,0];

                            fig = plt.figure();
                            fig.add_subplot(1,3,1);
                            plt.scatter(k_i, m_i);
                            plt.xlabel(r'$\phi_1$', fontsize=fontsize);
                            plt.ylabel(r'$\phi_2$', fontsize=fontsize);

                            fig.add_subplot(1,3,2);
                            plt.scatter(k_i, c_i);
                            plt.xlabel(r'$\phi_1$', fontsize=fontsize);
                            plt.ylabel(r'$\phi_3$', fontsize=fontsize);

                            fig.add_subplot(1,3,3);
                            plt.scatter(m_i, c_i);
                            plt.xlabel(r'$\phi_2$', fontsize=fontsize);
                            plt.ylabel(r'$\phi_3$', fontsize=fontsize);
                            plt.show();

                    elif (system.name == 'linear_1D'):
                        fig = plt.figure();
                        fig.add_subplot(1,2,1);
                        plt.hist(_phi[0,:,0,0]);
                        plt.title('phi');
                        fig.add_subplot(1,2,2);
                        plt.hist(_T_x[0,:,0]);
                        plt.title('T(g(phi))');
                        plt.show();

                    elif (system.name == 'linear_2D'):
                        data = pd.DataFrame(data=_phi[0,:,:,0]);
                        fig = plt.figure();
                        sns.pairplot(data);
                        plt.show();

                        _T_x_mean = np.mean(_T_x[0], 0);
                        plt.figure();
                        plt.subplot(1,2,1);
                        plt.plot(mu);
                        plt.plot(_T_x_mean)
                        plt.legend(['mu', 'E[T(x)]']);
                        plt.show();

                    elif (system.name == 'damped_harmonic_oscillator'):
                        fig = plt.figure();
                        print('system.D', system.D);
                        for plot_ind in range(system.D):
                            plt.subplot(1,system.D,plot_ind+1);
                            plt.hist(_phi[0,:,plot_ind,0]);
                            plt.title(r'$\phi_%d$' % (plot_ind+1));
                        plt.show();

                        fontsize = 14;
                        k_i = _phi[0,:,0,0];
                        m_i = _phi[0,:,1,0];
                        c_i = _phi[0,:,2,0];

                        fig = plt.figure();
                        fig.add_subplot(1,3,1);
                        plt.scatter(k_i, m_i);
                        plt.xlabel('k', fontsize=fontsize);
                        plt.ylabel('m', fontsize=fontsize);

                        fig.add_subplot(1,3,2);
                        plt.scatter(k_i, c_i);
                        plt.xlabel('k', fontsize=fontsize);
                        plt.ylabel('c', fontsize=fontsize);

                        fig.add_subplot(1,3,3);
                        plt.scatter(m_i, c_i);
                        plt.xlabel('m', fontsize=fontsize);
                        plt.ylabel('c', fontsize=fontsize);
                        plt.show();

                        _T_x_mean = np.mean(_T_x[0], 0);
                        plt.figure();
                        plt.subplot(1,2,1);
                        plt.plot(mu[:system.T]);
                        plt.plot(_T_x_mean[:system.T]);
                        plt.legend(['mu', 'E[T(x)]']);
                        for plot_ind in range(10):
                            plt.plot(_T_x[0,plot_ind,:system.T],'k--');
                        plt.title('X')

                        plt.subplot(1,2,2);
                        plt.plot(mu[system.T:]);
                        plt.plot(_T_x_mean[system.T:]);
                        plt.legend(['mu', 'E[T(x)]']);
                        plt.title('X^2')
                        plt.show();

                    z_i = np.random.normal(np.zeros((1,n,D,num_zi)), 1.0);
                    Hs[check_it] = _H;
                    costs[check_it] = cost_i;
                    T_x_mu_centereds[check_it] = np.mean(_T_x_mu_centered[0], 0);

                    iters = np.arange(check_rate, ((check_it+1)*check_rate)+1, check_rate);
                    plt.figure();
                    plt.subplot(1,2,1);
                    plt.plot(iters, costs[:(check_it+1)]);
                    plt.xlabel('iterations');
                    plt.ylabel('cost');
                    plt.subplot(1,2,2);
                    plt.plot(iters, Hs[:(check_it+1)]);
                    plt.xlabel('iterations');
                    plt.ylabel('H');
                    plt.show();
                    """
                    if stop_early:
                        has_converged = check_convergence([cost_grad_vals], cur_ind, cost_grad_lag, pthresh, criteria='grad_mean_ttest');
                    
                    if has_converged:
                        print('has converged!!!!!!');
                        convergence_it = cur_ind;
                    """

                    print('saving to %s  ...' % savedir);
                
                    np.savez(savedir + 'results.npz',  T_x_mu_centereds=T_x_mu_centereds, behavior=behavior, \
                                                       it=cur_ind, phi=_phi, \
                                                       convergence_it=convergence_it, check_rate=check_rate);
                
                    print(42*'*');
                    check_it += 1;

                sys.stdout.flush();
                i += 1;

            _R = np.mean(_T_x_mu_centered[0], 0)
            _lambda = _lambda + _c*_R;
            _c = 10*_c;
            total_its += i;


            # save all the hyperparams
            if not os.path.exists(savedir):
                print('Making directory %s' % savedir);
                os.makedirs(savedir);
            #saveParams(params, savedir);
            # save the model
            print('saving to', savedir);
            saver.save(sess, savedir + 'model');
    return costs, _phi, _T_x;
