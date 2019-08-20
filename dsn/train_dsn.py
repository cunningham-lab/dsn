#p Copyright 2018 Sean Bittner, Columbia University
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
import scipy.stats
import os
import sys
import io
from dsn.util.dsn_util import (
    setup_param_logging,
    initialize_adam_parameters,
    get_savedir,
    get_savestr,
    check_convergence,
)
from dsn.util.dsn_util import initialize_nf
from dsn.util.plot_util import make_training_movie

from tf_util.tf_util import (
    density_network,
    mixture_density_network,
    log_grads,
    AL_cost,
)
from tf_util.normalizing_flows import count_params
from tf_util.stat_util import sample_gumbel

def train_dsn(
    system,
    arch_dict,
    n=1000,
    AL_it_max=10,
    sigma_init=1.0,
    c_init_order=0,
    AL_fac=4.0,
    min_iters=1000,
    max_iters=5000,
    random_seed=0,
    lr_order=-3,
    check_rate=100,
    dir_str="general",
    savedir=None,
    entropy=True,
    db=False,
):
    """Trains a degenerate solution network (DSN).

        Args:
            system (obj): Instance of tf_util.systems.system.
            arch_dict (dict): Specifies structure of approximating density network.
            n (int): Batch size.
            AL_it_max (int): Number of augmented Lagrangian iterations.
            sigma_init (float): Gaussian initialization standard deviation.
            c_init_order (float): Augmented Lagrangian trade-off parameter initialization.
            min_iters (int): Minimum number of training iterations per AL epoch.
            max_iters (int): Maximum number of training iterations per AL epoch.
            random_seed (int): Tensorflow random seed for initialization.
            lr_order (float): Adam learning rate is 10^(lr_order).
            check_rate (int): Log diagonstics at every check_rate iterations.
            dir_str (str): Save directory name.
            entropy (bool): Include entropy in the cost function.
            db (bool): Record DSN samples on every diagnostic check.

        """
    # set initialization of AL parameter c and learning rate
    lr = 10 ** lr_order
    c_init = 10 ** c_init_order

    # save tensorboard summary in intervals
    TB_SAVE = False
    MODEL_SAVE = True
    TB_SAVE_EVERY = 50
    MODEL_SAVE_EVERY = 5000
    tb_save_params = False
    FIM = True
    MAX_TO_KEEP = 1000

    # Optimization hyperparameters:
    # If stop_early is true, test if parameter gradients over the last COST_GRAD_LAG
    # samples are significantly different than zero in each dimension.
    stop_early = False
    COST_GRAD_LAG = 100
    ALPHA = 0.05

    K = arch_dict['K']
    mixture = K > 1

    # Look for model initialization.  If not found, optimize the init.
    if (K > 1 and (not arch_dict['shared'])):
        initdirs = []
        for rs in range(1, K+1):
            print('Initializing %d/%d...' % (rs, K))
            initdirs.append(initialize_nf(system,
                                          arch_dict, 
                                          sigma_init,
                                          rs))
    else:
        print('Initializing...')
        initdirs = [initialize_nf(system,
                                  arch_dict, 
                                  sigma_init,
                                  random_seed)]


    print('initdirs ', initdirs)
    print('done.')

    # Reset tf graph, and set random seeds.
    tf.reset_default_graph()
    tf.set_random_seed(random_seed)

    # Load nf initialization
    W = tf.placeholder(tf.float64, shape=(None, None, system.D), name="W")

    # Create model save directory if doesn't exist.
    if (savedir is None):
        savedir = get_savedir(
            system, arch_dict, sigma_init, c_init_order, random_seed, dir_str
        )
    save_fname = savedir + 'opt_info.npz'
    param_fname = savedir + 'params.npz'

    if not os.path.exists(savedir):
        print("Making directory %s ." % savedir)
        os.makedirs(savedir)

    # Construct density network parameters.
    if (system.has_support_map):
        support_mapping = system.support_mapping
    else:
        support_mapping = None

    if (mixture):
        np.random.seed(random_seed)
        G = tf.placeholder(tf.float64, shape=(None, None, K), name="G")
        #Z, sum_log_det_jacobian, log_base_density, flow_layers, alpha, Mu, Sigma, C = mixture_density_network(
        Z, sum_log_det_jacobian, log_base_density, flow_layers, alpha, C = mixture_density_network(
            G, W, arch_dict, support_mapping, initdirs=initdirs
        )
    else: # mixture
        Z, sum_log_det_jacobian, flow_layers = density_network(
            W, arch_dict, support_mapping, initdir=initdirs[0]
        )

    # Permutations and batch norms
    # havent implemented mixture flows for real nvp archs yet
    if (not mixture):
        init_param_fname = initdirs[0] + 'theta.npz'
        init_param_file =  np.load(init_param_fname)
        init_thetas = init_param_file['theta'][()]

        final_thetas = {}
        batch_norm_mus = []
        batch_norm_sigmas = []
        batch_norm_layer_means = []
        batch_norm_layer_vars = []
        _batch_norm_mus = []
        _batch_norm_sigmas = []
        batch_norm = False
        for i in range(len(flow_layers)):
            flow_layer = flow_layers[i]
            if (flow_layer.name == 'PermutationFlow'):
                final_thetas.update({'DensityNetwork/Layer%d/perm_inds' % (i+1):flow_layer.inds})
            if (flow_layer.name == 'RealNVP' and flow_layer.batch_norm):
                batch_norm = True
                num_masks = arch_dict['real_nvp_arch']['num_masks']
                for j in range(num_masks):
                    batch_norm_mus.append(flow_layer.mus[j])
                    batch_norm_sigmas.append(flow_layer.sigmas[j])
                    batch_norm_layer_means.append(flow_layer.layer_means[j])
                    batch_norm_layer_vars.append(flow_layer.layer_vars[j])
                    _batch_norm_mus.append(init_thetas['DensityNetwork/batch_norm_mu%d' % (j+1)])
                    _batch_norm_sigmas.append(init_thetas['DensityNetwork/batch_norm_sigma%d' % (j+1)])


    with tf.name_scope("Entropy"):
        if (not mixture):
            log_base_density = tf.reduce_sum((-tf.square(W) / 2.0) - np.log(np.sqrt(2.0 * np.pi)), 2)
        log_q_z = log_base_density - sum_log_det_jacobian
        base_H = -tf.reduce_mean(log_base_density)
        sum_log_det_H = tf.reduce_mean(sum_log_det_jacobian)
        H = -tf.reduce_mean(log_q_z)
        tf.summary.scalar("H", H)

    all_params = tf.trainable_variables()
    nparams = len(all_params)

    with tf.name_scope("system"):
        # Compute system-specific sufficient statistics and log base measure on samples.
        T_x = system.compute_suff_stats(Z)
        mu = system.compute_mu()
        T_x_mu_centered = system.center_suff_stats_by_mu(T_x)
        if ('bounds' in system.behavior.keys()):
            I_x = system.compute_I_x(Z, T_x)
        else:
            I_x = None

    # Declare ugmented Lagrangian optimization hyperparameter placeholders.
    with tf.name_scope("AugLagCoeffs"):
        Lambda = tf.placeholder(dtype=tf.float64, shape=(system.num_suff_stats,))
        c = tf.placeholder(dtype=tf.float64, shape=())

    # Augmented Lagrangian cost function.
    print("Setting up augmented lagrangian gradient graph.")
    with tf.name_scope("AugLagCost"):
        cost, cost_grads, R_x = AL_cost(H, T_x_mu_centered, Lambda, c, \
                                      all_params, entropy=entropy, I_x=I_x)
        tf.summary.scalar("cost", cost)
        for i in range(system.num_suff_stats):
            tf.summary.scalar('R_%d' % (i+1), R_x[i])

    # Compute gradient of density network params (theta) wrt cost.
    grads_and_vars = []
    for i in range(len(all_params)):
        grads_and_vars.append((cost_grads[i], all_params[i]))

    # Compute inverse of dgm if known
    if (FIM and not mixture):
        if (arch_dict['flow_type'] == 'RealNVP'):
            print('computing inverse of realNVP')
            Z_INV = Z
            layer_ind = len(flow_layers) - 1
            while (layer_ind > -1):
                layer = flow_layers[layer_ind]
                Z_INV = layer.inverse(Z_INV)
                layer_ind -= 1

        else:
            Z_INV = tf.placeholder(tf.float64, (1,))


    # Add inputs and outputs of NF to saved tf model.
    tf.add_to_collection("W", W)
    tf.add_to_collection("Z", Z)
    tf.add_to_collection("log_q_z", log_q_z)
    if (FIM and not mixture):
        tf.add_to_collection("Z_INV", Z_INV)
    if (batch_norm):
        num_batch_norms = len(batch_norm_mus)
        for i in range(num_batch_norms):
            tf.add_to_collection("batch_norm_mu%d" % (i+1), batch_norm_mus[i])
            tf.add_to_collection("batch_norm_sigma%d" % (i+1), batch_norm_sigmas[i])
            tf.add_to_collection("batch_norm_layer_mean%d" % (i+1), batch_norm_layer_means[i])
            tf.add_to_collection("batch_norm_layer_var%d" % (i+1), batch_norm_layer_vars[i])

    saver = tf.train.Saver(max_to_keep=MAX_TO_KEEP)

    # Tensorboard logging
    summary_writer = tf.summary.FileWriter(savedir)
    if tb_save_params:
        setup_param_logging(all_params)

    summary_op = tf.summary.merge_all()

    config = tf.ConfigProto()
    # Allow the full trace to be stored at run time.
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

    num_diagnostic_checks = AL_it_max * (max_iters // check_rate) + 1
    nparam_vals = count_params(arch_dict)
    if (db):
        COST_GRAD_LOG_LEN = num_diagnostic_checks
        param_vals = np.zeros((COST_GRAD_LOG_LEN, nparam_vals))
    else:
        COST_GRAD_LOG_LEN = 2*COST_GRAD_LAG
        param_vals = None

    # Cyclically record gradients in a 2*COST_GRAD_LAG logger
    cost_grad_vals = np.zeros((COST_GRAD_LOG_LEN, nparam_vals))
    # Keep track of cost, entropy, and constraint violation throughout training.
    costs = np.zeros((num_diagnostic_checks,))
    Hs = np.zeros((num_diagnostic_checks,))
    base_Hs = np.zeros((num_diagnostic_checks,))
    sum_log_det_Hs = np.zeros((num_diagnostic_checks,))
    R2s = np.zeros((num_diagnostic_checks,))
    mean_T_xs = np.zeros((num_diagnostic_checks, system.num_suff_stats))

    # Keep track of AL parameters throughout training.
    cs = []
    lambdas = []
    epoch_inds = [0]

    # Take snapshots of z and log density throughout training.
    nsamps = n
    if (db):
        alphas = np.zeros((num_diagnostic_checks, K))
        mus = np.zeros((num_diagnostic_checks, K, system.D))
        sigmas = np.zeros((num_diagnostic_checks, K, system.D))
        Zs = np.zeros((num_diagnostic_checks, nsamps, system.D))
        Cs = np.zeros((num_diagnostic_checks, nsamps, K))
        log_q_zs = np.zeros((num_diagnostic_checks, nsamps))
        log_base_q_zs = np.zeros((num_diagnostic_checks, nsamps))
        T_xs = np.zeros((num_diagnostic_checks, nsamps, system.num_suff_stats))
        # params
        if (batch_norm):
            bn_mus = np.zeros((num_diagnostic_checks, num_batch_norms, system.D))
            bn_sigmas = np.zeros((num_diagnostic_checks, num_batch_norms, system.D))
    else:
        alphas = np.zeros((AL_it_max + 1, K))
        mus = np.zeros((AL_it_max + 1, K, system.D))
        sigmas = np.zeros((AL_it_max + 1, K, system.D))
        Zs = np.zeros((AL_it_max + 1, nsamps, system.D))
        Cs = np.zeros((AL_it_max + 1, nsamps, K))
        log_q_zs = np.zeros((AL_it_max + 1, nsamps))
        log_base_q_zs = np.zeros((AL_it_max + 1, nsamps))
        T_xs = np.zeros((AL_it_max + 1, nsamps, system.num_suff_stats))
        if (batch_norm):
            bn_mus = np.zeros((AL_it_max + 1, num_batch_norms, system.D))
            bn_sigmas = np.zeros((AL_it_max + 1, num_batch_norms, system.D))

    gamma = 0.25
    num_norms = 100
    norms = np.zeros((num_norms,))
    new_norms = np.zeros((num_norms,))

    np.random.seed(0)
    _c = c_init
    _lambda = np.zeros((system.num_suff_stats,))
    check_it = 0
    with tf.Session(config=config) as sess:
        print("training DSN for %s" % system.name)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        summary_writer.add_graph(sess.graph)

        # Log initial state of the DSN.
        w_i = np.random.normal(np.zeros((1, nsamps, system.D)), 1.0)
        feed_dict = {W: w_i, Lambda: _lambda, c: _c}

        # Initialize the batch norms.  Iteratively run out the coupling layers.
        if (batch_norm):
            print('Initializing batch norm parameters.')
            num_batch_norms = len(batch_norm_mus)
            for j in range(num_batch_norms):
                #_batch_norm_mus[j] = sess.run(batch_norm_layer_means[j], feed_dict)
                #_batch_norm_sigmas[j] = np.sqrt(sess.run(batch_norm_layer_vars[j], feed_dict))
                feed_dict.update({batch_norm_mus[j]:_batch_norm_mus[j]})
                feed_dict.update({batch_norm_sigmas[j]:_batch_norm_sigmas[j]})

        if (mixture):
            g_i = np.expand_dims(sample_gumbel(nsamps, K), 0)
            feed_dict.update({G: g_i})
       
        args = [cost, cost_grads, Z, T_x, H, base_H, sum_log_det_H, log_q_z, log_base_density, summary_op]
        if (batch_norm):
            args.append(batch_norm_layer_means)
            args.append(batch_norm_layer_vars)
        _args = sess.run(args, feed_dict)

        cost_i = _args[0]
        _cost_grads = _args[1]
        _Z = _args[2]
        _T_x = _args[3]
        _H = _args[4]
        _base_H = _args[5]
        _sld_H = _args[6]
        _log_q_z = _args[7]
        _log_base_q_z = _args[8]
        summary = _args[9]
        
        # Update batch norm params
        if (batch_norm):
            mom = 0.99
            _batch_norm_layer_means = _args[10]
            _batch_norm_layer_vars = _args[11]
            for j in range(num_batch_norms):
                _batch_norm_mus[j] = mom*_batch_norm_mus[j] + (1.0-mom)*_batch_norm_layer_means[j]
                _batch_norm_sigmas[j] = mom*_batch_norm_sigmas[j] + (1.0-mom)*np.sqrt(_batch_norm_layer_vars[j])

        summary_writer.add_summary(summary, 0)
        #log_grads(_cost_grads, cost_grad_vals, 0)
        if (db):
            _params = sess.run(all_params)
            #log_grads(_params, param_vals, 0)

        mean_T_xs[0, :] = np.mean(_T_x[0], 0)
        Hs[0] = _H
        base_Hs[0] = _base_H
        sum_log_det_Hs[0] = _sld_H
        costs[0] = cost_i
        check_it += 1

        if (mixture):
            #_alpha, _mu, _sigma, _C = sess.run([alpha, Mu, Sigma, C], {G:g_i})
            _alpha, _C = sess.run([alpha, C], {G:g_i})
            alphas[0,:] = _alpha
            #mus[0,:,:] = _mu
            #sigmas[0,:,:] = _sigma
            Cs[0,:,:] = _C
        Zs[0, :, :] = _Z[0, :, :]
        log_q_zs[0, :] = _log_q_z[0, :]
        log_base_q_zs[0, :] = _log_base_q_z[0, :]
        T_xs[0, :, :] = _T_x[0]

        bn_mus[0] = np.array(_batch_norm_mus)
        bn_sigmas[0] = np.array(_batch_norm_sigmas)

        if (MODEL_SAVE):
            print("Saving model at beginning.")
            saver.save(sess, savedir + "model", global_step=0)
            np.savez(
                    param_fname,
                    theta=final_thetas,
                    batch_norm_mus=bn_mus,
                    batch_norm_sigmas=bn_sigmas,
                )


        optimizer = tf.contrib.optimizer_v2.AdamOptimizer(learning_rate=lr)
        train_step = optimizer.apply_gradients(grads_and_vars)

        total_its = 1
        for k in range(AL_it_max):
            print("AL iteration %d" % (k + 1))
            cs.append(_c)
            lambdas.append(_lambda)

            # Reset the optimizer so momentum from previous epoch of AL optimization
            # does not effect optimization in the next epoch.
            initialize_adam_parameters(sess, optimizer, all_params)

            i = 0
            wrote_graph = False
            has_converged = False
            convergence_it = 0
            while i < max_iters:
                cur_ind = total_its + i

                w_i = np.random.normal(np.zeros((1, n, system.D)), 1.0)
                feed_dict.update({W: w_i})
                if (mixture):
                    g_i = np.expand_dims(sample_gumbel(n, K), 0)
                    feed_dict.update({G: g_i})

                # Log diagnostics for W draw before gradient step
                if np.mod(cur_ind + 1, check_rate) == 0:
                    _H, _base_H, _sld_H, _T_x, _Z, _log_q_z, _log_base_q_z = \
                        sess.run([H, base_H, sum_log_det_H, T_x, Z, log_q_z, log_base_density], feed_dict)
                    print(42 * "*")
                    print("it = %d " % (cur_ind + 1))
                    if (mixture):
                        _alpha = sess.run(alpha)
                        print('alpha', _alpha)
                    print("H", _H, "cost", cost_i)
                    sys.stdout.flush()
                    
                    Hs[check_it] = _H
                    base_Hs[check_it] = _base_H
                    sum_log_det_Hs[check_it] = _sld_H
                    mean_T_xs[check_it] = np.mean(_T_x[0], 0)

                    if (db):
                        if (mixture):
                            #_alpha, _mu, _sigma, _C = sess.run([alpha, Mu, Sigma, C], {G:g_i})
                            _alpha, _C = sess.run([alpha, C], {G:g_i})
                            alphas[check_it,:] = _alpha
                            #mus[check_it,:,:] = _mu
                            #sigmas[check_it,:,:] = _sigma
                            Cs[check_it,:,:] = _C
                        Zs[check_it, :, :] = _Z[0, :, :]
                        log_q_zs[check_it, :] = _log_q_z[0, :]
                        log_base_q_zs[check_it, :] = _log_base_q_z[0, :]
                        T_xs[check_it, :, :] = _T_x[0]

                        bn_mus[check_it] = np.array(_batch_norm_mus)
                        bn_sigmas[check_it] = np.array(_batch_norm_sigmas)

                        if (MODEL_SAVE):
                            print("Saving model at iter %d." % cur_ind)
                            saver.save(sess, savedir + "model", global_step=check_it)
                            np.savez(
                                    param_fname,
                                    theta=final_thetas,
                                    batch_norm_mus=bn_mus,
                                    batch_norm_sigmas=bn_sigmas,
                                )


                    if stop_early:
                        has_converged = check_convergence(
                            cost_grad_vals, cur_ind % COST_GRAD_LOG_LEN, COST_GRAD_LAG, ALPHA
                        )

                    if has_converged:
                        print("has converged!!!!!!")
                        sys.stdout.flush()
                        convergence_it = cur_ind
                        break

                    np.savez(
                        save_fname,
                        costs=costs,
                        cost_grad_vals=cost_grad_vals,
                        param_vals=param_vals,
                        Hs=Hs,
                        base_Hs=base_Hs,
                        sum_log_det_Hs=sum_log_det_Hs,
                        R2s=R2s,
                        mean_T_xs=mean_T_xs,
                        fixed_params=system.fixed_params,
                        behavior=system.behavior,
                        mu=system.mu,
                        it=cur_ind,
                        alphas=alphas,
                        mus=mus,
                        sigmas=sigmas,
                        Cs=Cs,
                        Zs=Zs,
                        cs=cs,
                        lambdas=lambdas,
                        log_q_zs=log_q_zs,
                        log_base_q_zs=log_base_q_zs,
                        T_xs=T_xs,
                        convergence_it=convergence_it,
                        check_rate=check_rate,
                        epoch_inds=epoch_inds,
                        n=n,
                        sigma_init=sigma_init,
                        c_init_order=c_init_order,
                        AL_fac=AL_fac,
                        min_iters=min_iters,
                        max_iters=max_iters,
                    )

                    print(42 * "*")

                if np.mod(cur_ind, check_rate) == 0:
                    start_time = time.time()

                if (TB_SAVE and np.mod(cur_ind, TB_SAVE_EVERY) == 0):
                    # Create a fresh metadata object:
                    run_metadata = tf.RunMetadata()
                    ts, cost_i, _cost_grads, summary = sess.run([train_step, cost, cost_grads, summary_op], 
                                       feed_dict,
                                       options=run_options,
                                       run_metadata=run_metadata)
                    summary_writer.add_summary(summary, cur_ind)
                    if (not wrote_graph and i>20): # In case a GPU needs to warm up for optims
                        assert(min_iters >= 20 and TB_SAVE_EVERY >= 20)
                        print("Writing graph stuff for AL iteration %d." % (k+1))
                        summary_writer.add_run_metadata(run_metadata, 
                                                        "train_step_{}".format(cur_ind),
                                                        cur_ind)
                        wrote_graph = True
                else:
                    ts, cost_i, _cost_grads = sess.run([train_step, cost, cost_grads], feed_dict)

                if np.mod(cur_ind + 1, check_rate) == 0:
                    costs[check_it] = cost_i
                    check_it += 1

                if np.mod(cur_ind, check_rate) == 0:
                    end_time = time.time()
                    print("Iteration took %.4f seconds." % (end_time - start_time))

                #log_grads(_cost_grads, cost_grad_vals, cur_ind % COST_GRAD_LOG_LEN)

                if (batch_norm):
                    for j in range(num_batch_norms):
                        feed_dict.update({batch_norm_mus[j]:_batch_norm_mus[j]})
                        feed_dict.update({batch_norm_sigmas[j]:_batch_norm_sigmas[j]})

                sys.stdout.flush()
                i += 1
            w_k = np.random.normal(np.zeros((1, nsamps, system.D)), 1.0)
            feed_dict.update({W: w_k})
            if (mixture):
                g_k = np.expand_dims(sample_gumbel(nsamps, K), 0)
                feed_dict.update({G: g_k})
            _H, _T_x, _Z, _log_q_z, _log_base_q_z = sess.run([H, T_x, Z, log_q_z, log_base_density], feed_dict)

            if (not db):
                if (mixture):
                    #_alpha, _mu, _sigma, _C = sess.run([alpha, Mu, Sigma, C], {G:g_i})
                    _alpha, _C = sess.run([alpha, C], {G:g_i})
                    alphas[k+1,:] = _alpha
                    #mus[k+1,:] = _mu
                    #sigmas[k+1,:] = _sigma
                    Cs[k+1,:,:] = _C
                Zs[k + 1, :, :] = _Z[0, :, :]
                log_q_zs[k + 1, :] = _log_q_z[0, :]
                log_base_q_zs[k + 1, :] = _log_base_q_z[0, :]
                T_xs[k + 1, :, :] = _T_x[0]
            _T_x_mu_centered = sess.run(T_x_mu_centered, feed_dict)
            _R = np.mean(_T_x_mu_centered[0], 0)
            _lambda = _lambda + _c * _R

            # save all the hyperparams
            if not os.path.exists(savedir):
                print("Making directory %s" % savedir)
                os.makedirs(savedir)
            # saveParams(params, savedir);
            # save the model
            print("saving to", savedir)
            if (MODEL_SAVE and not db):
                saver.save(sess, savedir + "model", global_step=k)
                np.savez(
                        param_fname,
                        theta=final_thetas,
                        batch_norm_mus=bn_mus,
                        batch_norm_sigmas=bn_sigmas,
                    )

            total_its += i
            epoch_inds.append(total_its - 1)


            # If optimizing for feasible set and on f.s., quit.
            if (system.behavior["type"] == "feasible"):
                is_feasible = system.behavior["is_feasible"](_T_x[0])
                if is_feasible:
                    print('On the feasible set.  Initialization complete.')
                    break
                else:
                    print('Not on safe part of feasible set yet.')

            # do the hypothesis test to figure out whether or not we should update c
            for j in range(num_norms):
                w_j = np.random.normal(np.zeros((1, n, system.D)), 1.0)
                feed_dict.update({W: w_j})
                if (mixture):
                    g_j = np.expand_dims(sample_gumbel(n, K), 0)
                    feed_dict.update({G: g_j})
                _T_x_mu_centered = sess.run(T_x_mu_centered, feed_dict)
                _R = np.mean(_T_x_mu_centered[0], 0)
                new_norms[j] = np.linalg.norm(_R)

            t, p = scipy.stats.ttest_ind(new_norms, gamma * norms, equal_var=False)
            # probabilistic update based on p value
            u = np.random.rand(1)
            print("t", t, "p", p)
            if u < 1 - p / 2.0 and t > 0:
                print(u, "not enough! c updated")
                _c = AL_fac * _c
            else:
                print(u, "same c")

            feed_dict.update({Lambda: _lambda, c: _c})

            norms = new_norms

        for i in range(nparams):
            final_thetas.update({all_params[i].name:sess.run(all_params[i])});


        if (MODEL_SAVE):
            print("Saving model before exit")
            if db:
                global_step = check_it
            else:
                global_step = k
            saver.save(sess, savedir + "model", global_step=global_step)
            np.savez(
                    param_fname,
                    theta=final_thetas,
                    batch_norm_mus=bn_mus,
                    batch_norm_sigmas=bn_sigmas,
                )

    print("saving to %s  ..." % savedir)
    sys.stdout.flush()
                    
    np.savez(
        save_fname,
        costs=costs,
        cost_grad_vals=cost_grad_vals,
        param_vals=param_vals,
        Hs=Hs,
        base_Hs=base_Hs,
        sum_log_det_Hs=sum_log_det_Hs,
        R2s=R2s,
        mean_T_xs=mean_T_xs,
        fixed_params=system.fixed_params,
        behavior=system.behavior,
        mu=system.mu,
        it=cur_ind,
        alphas=alphas,
        mus=mus,
        sigmas=sigmas,
        Cs=Cs,
        Zs=Zs,
        cs=cs,
        lambdas=lambdas,
        log_q_zs=log_q_zs,
        log_base_q_zs=log_base_q_zs,
        T_xs=T_xs,
        convergence_it=convergence_it,
        check_rate=check_rate,
        epoch_inds=epoch_inds,
        n=n,
        sigma_init=sigma_init,
        c_init_order=c_init_order,
        AL_fac=AL_fac,
        min_iters=min_iters,
        max_iters=max_iters,
    )

    # make training movie
    step = 1
    video_fname = savedir + get_savestr(system, arch_dict, sigma_init, c_init_order, random_seed) + "_video"
    make_training_movie(savedir, system, step, save_fname=video_fname)

    if (system.behavior["type"] == "feasible"):
        return costs, _Z, is_feasible
    else:
        return costs, _Z


