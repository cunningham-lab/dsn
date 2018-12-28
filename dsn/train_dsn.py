# Copyright 2018 Sean Bittner, Columbia University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
import tensorflow as tf
import numpy as np
import time
import csv
from datetime import datetime
import scipy.stats
import sys
import os
import datetime
import io 
from sklearn.metrics import pairwise_distances
import pandas as pd
do_plot = False;
if (do_plot):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
from dsn.util.dsn_util import time_invariant_flow, check_convergence, setup_param_logging, \
                      initialize_optimization_parameters, computeMoments, getEtas, \
                      approxKL, setup_IO, get_initdir, compute_R2, AR_to_autocov_np, \
                      log_grads
from tf_util.tf_util import construct_density_network, declare_theta, \
                                connect_density_network, count_params, AL_cost, \
                                memory_extension, load_nf_init
from tf_util.families import family_from_str
from efn.train_nf import train_nf

def train_dsn(system, behavior, n, flow_dict, k_max=10, sigma_init=10.0, c_init_order=0, lr_order=-3, \
              random_seed=0, min_iters=1000,  max_iters=5000, check_rate=100, \
              dir_str='general'):
    """Trains a degenerate solution network (DSN).

        Args:
            system (obj): Instance of tf_util.systems.system.
            behavior (dict): Mean parameters of model behavior to learn with DSN.
            n (int): Batch size.
            flow_dict (dict): Specifies structure of approximating density network.
            k_max (int): Number of augmented Lagrangian iterations.
            c_init (float): Augmented Lagrangian trade-off parameter initialization.
            lr_order (float): Adam learning rate is 10^(lr_order).
            check_rate (int): Log diagonstics at every check_rate iterations.
            max_iters (int): Maximum number of training iterations.
            random_seed (int): Tensorflow random seed for initialization.

        """
    # Learn a single (K=1) distribution with a DSN.
    K = 1;

    # Since optimization may converge early, we dynamically allocate space to record
    # model diagnostics as optimization progresses.
    OPT_COMPRESS_FAC = 128

    # set initialization of AL parameter c and learning rate
    lr = 10**lr_order
    c_init = 10**c_init_order;

    # Reset tf graph, and set random seeds.
    tf.reset_default_graph();
    tf.set_random_seed(random_seed);
    np.random.seed(0);

    # save tensorboard summary in intervals
    TB_SAVE_EVERY = 50;
    MODEL_SAVE_EVERY = 5000;
    tb_save_params = False;

    # Optimization hyperparameters:
    # If stop_early is true, test if parameter gradients over the last COST_GRAD_LAG
    # samples are significantly different than zero in each dimension.
    stop_early = False;
    COST_GRAD_LAG = 100;
    P_THRESH = 0.05;

    # Look for model initialization
    initdir = initialize_nf(system.D, flow_dict, sigma_init, random_seed)
    print(initdir);
    inits = load_nf_init(initdir, flow_dict);
    flow_dict.update({"inits":inits});

    W = tf.placeholder(tf.float64, shape=(None, None, system.D, None), name="W")
    p0 = tf.reduce_prod(tf.exp((-tf.square(W)) / 2.0) / np.sqrt(2.0 * np.pi), axis=[2, 3]);
    base_log_q_z = tf.log(p0[:, :]);

    flow_layers, num_theta_params = construct_density_network(flow_dict, system.D, 1);
    flow_layers, num_theta_params = system.map_to_parameter_support(flow_layers, num_theta_params);


    # Create model save directory if doesn't exist.
    savedir = setup_IO(system, flow_dict, sigma_init, lr_order, c_init_order, random_seed, dir_str);
    if not os.path.exists(savedir):
        print('Making directory %s' % savedir );
        os.makedirs(savedir);

    # Declare density network parameters.
    theta = declare_theta(flow_layers, inits);

    # Connect declared tf Variables theta to the density network.
    z, sum_log_det_jacobian, Z_by_layer = connect_density_network(W, flow_layers, theta);
    log_q_z = base_log_q_z - sum_log_det_jacobian;

    all_params = tf.trainable_variables()
    nparams = len(all_params)

    # Compute family-specific sufficient statistics and log base measure on samples.
    T_x = system.compute_suff_stats(z);
    mu = system.compute_mu(behavior);
    T_x_mu_centered = system.center_suff_stats_by_mu(T_x, mu);

    R2 = compute_R2(log_q_z, None, T_x);

    # Declare ugmented Lagrangian optimization hyperparameter placeholders.
    Lambda = tf.placeholder(dtype=tf.float64, shape=(system.num_suff_stats,));
    c = tf.placeholder(dtype=tf.float64, shape=());

    # Augmented Lagrangian cost function.
    cost, cost_grads, H = AL_cost(log_q_z, T_x_mu_centered, Lambda, c, all_params);

    # Compute gradient of density network params (theta) wrt cost.
    grads_and_vars = [];
    for i in range(len(all_params)):
        grads_and_vars.append((cost_grads[i], all_params[i]));

    # Add inputs and outputs of NF to saved tf model.
    tf.add_to_collection('W', W);
    tf.add_to_collection('z', z);
    saver = tf.train.Saver();

    # Tensorboard logging.
    summary_writer = tf.summary.FileWriter(savedir);
    tf.summary.scalar('H', -tf.reduce_mean(log_q_z));
    tf.summary.scalar('cost', cost);
    if (tb_save_params):
        setup_param_logging(all_params);

    summary_op = tf.summary.merge_all();

    # Dynamically extend logging of parameter gradients.
    array_init_len = int(np.ceil((max_iters*k_max)/OPT_COMPRESS_FAC));
    nparam_vals = count_params(all_params);
    cost_grad_vals = np.zeros((array_init_len, nparam_vals));
    array_cur_len = array_init_len;

    # Keep track of cost, entropy, and constraint violation throughout training.
    num_diagnostic_checks = k_max*(max_iters // check_rate)+1;
    costs = np.zeros((num_diagnostic_checks,));
    Hs = np.zeros((num_diagnostic_checks,));
    R2s = np.zeros((num_diagnostic_checks,));
    mean_T_xs = np.zeros((num_diagnostic_checks, system.num_suff_stats));

    # Keep track of AL parameters throughout training.
    cs = [];
    lambdas = [];

    # Take snapshots of z and log density throughout training.
    zs = np.zeros((k_max+1, n, system.D));
    log_q_zs = np.zeros((k_max+1, n));
    T_xs = np.zeros((k_max+1, n, system.num_suff_stats));

    gamma = 0.25;
    num_norms = 100;
    norms = np.zeros((num_norms,));
    new_norms = np.zeros((num_norms,));

    _c = c_init;
    _lambda = np.zeros((system.num_suff_stats,));
    check_it = 0;
    with tf.Session() as sess:
        print('training DSN for %s' % system.name);
        init_op = tf.global_variables_initializer();
        sess.run(init_op);

        # Log initial state of the DSN.
        w_i = np.random.normal(np.zeros((K,n,system.D,1)), 1.0);
        feed_dict = {W:w_i, Lambda:_lambda, c:_c};
        cost_i, _cost_grads, _z, _T_x, _H, _log_q_z, summary = \
            sess.run([cost, cost_grads, z, T_x, H, log_q_z, summary_op], feed_dict);
        

        summary_writer.add_summary(summary, 0);
        log_grads(_cost_grads, cost_grad_vals, 0);

        mean_T_xs[0,:] = np.mean(_T_x[0], 0);
        Hs[0] = _H;
        #R2s[0] = _R2;
        costs[0] = cost_i;
        check_it += 1;

        zs[0,:,:] = _z[0,:,:,0];
        log_q_zs[0,:] = _log_q_z[0,:];
        T_xs[0,:,:] = _T_x[0];
        
       
        total_its = 1;
        for k in range(k_max):
            print("AL iteration %d" % (k+1));
            cs.append(_c);
            lambdas.append(_lambda);

            # Reset the optimizer so momentum from previous epoch of AL optimization
            # does not effect optimization in the next epoch. 
            print('resetting optimizer');
            optimizer = tf.contrib.optimizer_v2.AdamOptimizer(learning_rate=lr);
            train_step = optimizer.apply_gradients(grads_and_vars);
            initialize_optimization_parameters(sess, optimizer, all_params);

            for j in range(num_norms):
                w_j = np.random.normal(np.zeros((1,n,system.D,1)), 1.0);
                feed_dict.update({W:w_j});
                _T_x_mu_centered = sess.run(T_x_mu_centered, feed_dict);
                _R = np.mean(_T_x_mu_centered[0], 0)
                norms[j] = np.linalg.norm(_R);

            i = 0;
            has_converged = False;
            convergence_it = 0;
            print('Aug Lag it', k);
            if (do_plot):
                print('lambda', _lambda);
                print('c', _c);

            while (i < max_iters):
                cur_ind = total_its + i;

                if (cur_ind == array_cur_len):
                    cost_grad_vals = memory_extension([cost_grad_vals], array_cur_len)[0];
                    array_cur_len = 2*array_cur_len;

                w_i = np.random.normal(np.zeros((K,n,system.D,1)), 1.0);
                feed_dict = {W:w_i, Lambda:_lambda, c:_c};

                start_time = time.time();
                ts, cost_i, _cost_grads, summary, _T_x, _H, _z = \
                    sess.run([train_step, cost, cost_grads, summary_op, T_x, H, z], feed_dict);
                end_time = time.time();
                if (np.mod(cur_ind, check_rate)==0):
                    print('iteration took %.4f seconds.' % (end_time-start_time));
                    
                log_grads(_cost_grads, cost_grad_vals, cur_ind);

                if (np.mod(cur_ind, TB_SAVE_EVERY)==0):
                    summary_writer.add_summary(summary, cur_ind);

                if (np.mod(i, MODEL_SAVE_EVERY) == 0):
                    print('saving model at iter', i);
                    saver.save(sess, savedir + 'model');

                if (np.mod(cur_ind+1, check_rate)==0):
                    _H, _T_x, _z, _log_q_z = sess.run([H, T_x, z, log_q_z], feed_dict);
                    print(42*'*');
                    print('it = %d ' % (cur_ind+1));
                    print('H', _H);
                    #print('R2', _R2);
                    print('cost', cost_i);
                    sys.stdout.flush();

                    Hs[check_it] = _H;
                    #R2s[check_it] = _R2;
                    costs[check_it] = cost_i;
                    mean_T_xs[check_it] = np.mean(_T_x[0], 0);

                    if stop_early:
                        has_converged = check_convergence([cost_grad_vals], cur_ind, COST_GRAD_LAG, P_THRESH, criteria='grad_mean_ttest');
                    
                    if has_converged:
                        print('has converged!!!!!!');
                        convergence_it = cur_ind;
                        break;

                    print('saving to %s  ...' % savedir);
                
                    np.savez(savedir + 'results.npz',  costs=costs, Hs=Hs, R2s=R2s, mean_T_xs=mean_T_xs, behavior=behavior, mu=mu, \
                                                       it=cur_ind, zs=zs, cs=cs, lambdas=lambdas, log_q_zs=log_q_zs,  \
                                                        T_xs=T_xs, convergence_it=convergence_it, check_rate=check_rate);
                
                    print(42*'*');
                    check_it += 1;

                sys.stdout.flush();
                i += 1;
            zs[k+1,:,:] = _z[0,:,:,0];
            log_q_zs[k+1,:] = _log_q_z[0,:];
            T_xs[k+1,:,:] = _T_x[0];
            _T_x_mu_centered = sess.run(T_x_mu_centered, feed_dict);
            _R = np.mean(_T_x_mu_centered[0], 0)
            _lambda = _lambda + _c*_R;

            # do the hypothesis test to figure out whether or not we should update c
            feed_dict = {Lambda:_lambda, c:_c};

            for j in range(num_norms):
                w_j = np.random.normal(np.zeros((1,n,system.D,1)), 1.0);
                feed_dict.update({W:w_j});
                _T_x_mu_centered = sess.run(T_x_mu_centered, feed_dict);
                _R = np.mean(_T_x_mu_centered[0], 0)
                new_norms[j] = np.linalg.norm(_R);

            t,p = scipy.stats.ttest_ind(new_norms, gamma*norms, equal_var = False)
            # probabilistic update based on p value
            u = np.random.rand(1)
            print('t', t, 'p', p);
            if u < 1-p/2.0 and t>0:
                print(u, 'not enough! c updated');
                _c = 4*_c;
            else:
                print(u, 'same c');

            _c = 4*_c;
            total_its += i;


            # save all the hyperparams
            if not os.path.exists(savedir):
                print('Making directory %s' % savedir);
                os.makedirs(savedir);
            #saveParams(params, savedir);
            # save the model
            print('saving to', savedir);
            saver.save(sess, savedir + 'model');
    np.savez(savedir + 'results.npz',  costs=costs, Hs=Hs, R2s=R2s, mean_T_xs=mean_T_xs, behavior=behavior, mu=mu, \
                                       it=cur_ind, zs=zs, cs=cs, lambdas=lambdas, log_q_zs=log_q_zs, \
                                       T_xs=T_xs, convergence_it=convergence_it, check_rate=check_rate);
    return costs, _z, _T_z;

def initialize_nf(D, flow_dict, sigma_init, random_seed, min_iters=50000):
    initdir = get_initdir(D, flow_dict, sigma_init, random_seed)
    print('initdir', initdir);
    initfname = initdir + 'final_theta.npz';
    resfname = initdir + 'results.npz';

    learn_init = True;
    if os.path.exists(initfname):

        resfile = np.load(resfname);
        assert(resfile['converged']);
    
    else:
        print(initfname);
        print('training NF');
        fam_class = family_from_str('normal');
        family = fam_class(D);
        params = {'mu':np.zeros((D,)), \
                  'Sigma':np.square(sigma_init)*np.eye(D), \
                  'dist_seed':0};
        n = 1000;
        lr_order = -3;
        check_rate = 100;
        max_iters = 1000000;
        train_nf(family, params, flow_dict, n, lr_order, random_seed, \
                 min_iters, max_iters, check_rate, None, profile=False, \
                 savedir=initdir);
        print('done initializing NF');
    return initdir;
