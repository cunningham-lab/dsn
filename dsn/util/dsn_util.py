import tensorflow as tf
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_kernels
from scipy.stats import ttest_1samp, multivariate_normal
import matplotlib.pyplot as plt
from tf_util.tf_util import get_archstring, get_initdir, check_init
import scipy.linalg
from dsn.util.systems import V1Circuit, SCCircuit, STGCircuit
from dsn.util.plot_util import assess_constraints_mix

from tf_util.families import family_from_str
from efn.train_nf import train_nf


def get_savedir(
    system, arch_dict, sigma_init, c_init_order, random_seed, dir_str, randsearch=False
):
    # set file I/O stuff
    resdir = "models/" + dir_str + "/"
    savestr = get_savestr(system, arch_dict, sigma_init, c_init_order, random_seed, randsearch)
    savedir = resdir + savestr + "/"
    return savedir

def get_savestr(system, arch_dict, sigma_init, c_init_order, random_seed, randsearch=False):
    archstring = get_archstring(arch_dict)
    sysparams = system.free_params[0]
    num_free_params = len(system.free_params)
    if num_free_params > 1:
        if (num_free_params >= 5):
            sysparams = "D=%d" % system.D
        else:
            for i in range(1, num_free_params):
                sysparams += "_%s" % system.free_params[i]

    if (randsearch):
        savestr = "%s_%s_%s_flow=%s_rs=%d" % (
                    system.name,
                    sysparams,
                    system.behavior_str,
                    archstring,
                    random_seed,
                    )
    else:
        savestr = "%s_%s_%s_flow=%s_sigma=%.2f_c=%d_rs=%d" % (
                    system.name,
                    sysparams,
                    system.behavior_str,
                    archstring,
                    sigma_init,
                    c_init_order,
                    random_seed,
                    )
    return savestr

def get_system_from_template(sysname, param_dict):
    if (sysname == "V1Circuit"):
        behavior_type = param_dict["behavior_type"]
        if (behavior_type == "difference"):
            base_I = 0.15
            run_I = 0.3
            tau = 0.15
            s = param_dict["s"]
            fac = param_dict["fac"]
            fixed_params = {'b_E':base_I, \
                            'b_P':base_I, \
                            'b_S':base_I, \
                            'b_V':base_I, \
                            'h_RUNE':run_I, \
                            'h_RUNP':run_I, \
                            'h_RUNS':run_I, \
                            'h_RUNV':run_I, \
                            'h_FFE':0.0, \
                            'h_FFP':0.0, \
                            'h_LATE':0.0, \
                            'h_LATP':0.0, \
                            'h_LATS':0.0, \
                            'h_LATV':0.0, \
                            'tau':tau, \
                            'n':2.0, \
                            's_0':30};
            behavior_type = "difference"
            c_vals=np.array([1.0])
            s_vals=np.array([s])
            r_vals=np.array([0.0, 1.0])
            C = c_vals.shape[0]*s_vals.shape[0]*r_vals.shape[0]
            bounds = np.zeros((C*4,))
            behavior = {'type':behavior_type, \
                        'c_vals':c_vals, \
                        's_vals':s_vals, \
                        'r_vals':r_vals, \
                        'fac':fac,
                        'bounds':bounds}
            model_opts = {"g_FF": "c", "g_LAT": "square", "g_RUN": "r"}
            T = 50
            dt = 0.05
            init_conds = np.expand_dims(np.array([1.0, 1.1, 1.2, 1.3]), 1)
            system = V1Circuit(fixed_params, behavior, model_opts, T, dt, init_conds)
        else:
            raise NotImplementedError()
        
    elif (sysname == 'SCCircuit'):
        behavior_type = param_dict["behavior_type"]
        fixed_params = {'E_constant':0.0, \
            'E_Pbias':0.1, \
            'E_Prule':0.5, \
            'E_Arule':0.5, \
            'E_choice':-0.2, \
            'E_light':0.1};
        if behavior_type == "WTA":
            #C = 1
            C = 2
            param_str = "full"
            p = param_dict['p']
            inact_str = param_dict['inact_str']
            means = np.array([p, p, 0.0, 0.0, 1.0, 1.0])
            #means = np.array([p, 0.0, 1.0])
            barrier_EPS = 1e-10
            if (p==0.0 or p==1.0):
                behavior = {
                    "type": behavior_type,
                    "means": means,
                    "inact_str":inact_str
                }
            else:
                behavior = {
                    "type": behavior_type,
                    "means": means,
                    "bounds":np.zeros(C) - barrier_EPS,
                    "inact_str":inact_str
                }
            model_opts = {"params":param_str, "C":C}
            system = SCCircuit(fixed_params, behavior, model_opts)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    return system

def get_ME_model(system, arch_dict, c_init_ords, sigma_inits, random_seeds, dirstr, conv_dict):
    num_sigmas = sigma_inits.shape[0]
    num_cs = c_init_ords.shape[0]
    num_rs = random_seeds.shape[0]
    
    model_dirs = []
    for i in range(num_cs):
        c_init_order = c_init_ords[i]
        for j in range(num_sigmas):
            sigma_init = sigma_inits[j]
            for k in range(num_rs):
                rs = random_seeds[k]
                savedir = get_savedir(system, arch_dict, sigma_init, c_init_order, rs, dirstr)
                model_dirs.append(savedir)
                
    first_its, ME_its, MEs = assess_constraints_mix(model_dirs, 
                                                    tol=conv_dict['tol'], 
                                                    tol_inds=conv_dict['tol_inds'],
                                                    alpha=conv_dict['alpha'], 
                                                    frac_samps=conv_dict['frac_samples']
                                                    )
    num_models = len(model_dirs)
    # Find the model that had maximum entropy while satisfying convergence criteria
    # iterate because of Nones
    flg = True
    for i in range(num_models):
        if (flg):
            if MEs[i] is not None:
                max_ME = MEs[i]
                max_ind = i
                flg = False
        else:
            if (MEs[i] is not None) and MEs[i] > max_ME:
                max_ME = MEs[i]
                max_ind = i
    
    if (flg):
        best_model = None
        max_ME = None
        first_it = None
        ME_it = None
    else:
        best_model = model_dirs[max_ind]
        ME_it = ME_its[max_ind]
        first_it = first_its[max_ind]
    
    return best_model, max_ME, ME_it, first_it
    
    
                

def initialize_adam_parameters(sess, optimizer, all_params):
    nparams = len(all_params)
    slot_names = optimizer.get_slot_names()
    num_slot_names = len(slot_names)
    for i in range(nparams):
        param = all_params[i]
        for j in range(num_slot_names):
            slot_name = slot_names[j]
            opt_var = optimizer.get_slot(param, slot_name)
            sess.run(opt_var.initializer)
    # Adam optimizer is the only tf optimizer that has this slot variable issue
    # href https://github.com/tensorflow/tensorflow/issues/8057
    if isinstance(optimizer, tf.contrib.optimizer_v2.AdamOptimizer):
        beta1_power, beta2_power = optimizer._get_beta_accumulators()
        sess.run(beta1_power.initializer)
        sess.run(beta2_power.initializer)
    return None


def check_convergence(cost_grad_vals, cur_ind, lag, alpha):
    logger_len = cost_grad_vals.shape[0]
    if (cur_ind < lag):
        last_grads = np.concatenate((cost_grad_vals[-(lag-cur_ind):, :], cost_grad_vals[:cur_ind, :]), 0)
    else:
        last_grads = cost_grad_vals[(cur_ind - lag) : cur_ind, :]
    nvars = last_grads.shape[1]
    has_converged = True
    for i in range(nvars):
        t, p = ttest_1samp(last_grads[:, i], 0)
        # if any grad mean is not zero, reject
        if p < (alpha / nvars):
            has_converged = False
            break
    return has_converged


def compute_R2(log_q_x, log_h_x, T_x_in):
    T_x_shape = tf.shape(T_x_in)
    M = T_x_shape[1]
    num_suff_stats = T_x_shape[2]

    T_x_in = T_x_in[0] - tf.expand_dims(tf.reduce_mean(T_x_in[0], 0), 0)
    T_x = tf.concat((T_x_in, tf.ones((M, 1), tf.float64)), axis=1)
    prec_mat = tf.matmul(tf.transpose(T_x), T_x)
    # + c*tf.eye(num_suff_stats+1, dtype=tf.float64);

    if log_h_x is not None:
        y_q = tf.expand_dims(log_q_x[0] - log_h_x[0], 1)
    else:
        y_q = tf.expand_dims(log_q_x[0], 1)

    beta_q = tf.matmul(tf.matrix_inverse(prec_mat), tf.matmul(tf.transpose(T_x), y_q))

    # compute optimial linear regression offset term for eta
    residuals_q = y_q - tf.matmul(T_x, beta_q)
    RSS = tf.matmul(tf.transpose(residuals_q), residuals_q)
    y_q_mc = y_q - tf.reduce_mean(y_q)
    TSS = tf.matmul(tf.transpose(y_q_mc), y_q_mc)
    # compute the R^2 of the exponential family fit
    R2_q = 1.0 - (RSS[0, 0] / TSS[0, 0])
    return R2_q


def gradients(f, x, grad_ys=None):
    """
    An easier way of computing gradients in tensorflow. The difference from tf.gradients is
        * If f is not connected with x in the graph, it will output 0s instead of Nones. This will be more meaningful
            for computing higher-order gradients.
        * The output will have the same shape and type as x. If x is a list, it will be a list. If x is a Tensor, it
            will be a tensor as well.
    :param f: A `Tensor` or a list of tensors to be differentiated
    :param x: A `Tensor` or a list of tensors to be used for differentiation
    :param grad_ys: Optional. It is a `Tensor` or a list of tensors having exactly the same shape and type as `f` and
                    holds gradients computed for each of `f`.
    :return: A `Tensor` or a list of tensors having the same shape and type as `x`

    got this func from https://gist.github.com/yang-song/07392ed7d57a92a87968e774aef96762
    """

    if isinstance(x, list):
        grad = tf.gradients(f, x, grad_ys=grad_ys)
        for i in range(len(x)):
            if grad[i] is None:
                grad[i] = tf.zeros_like(x[i])
        return grad
    else:
        grad = tf.gradients(f, xad, grad_ys=grad_ys)[0]
        if grad is None:
            return tf.zeros_like(x)
        else:
            return grad


def Lop(f, x, v):
    """
    Compute Jacobian-vector product. The result is v^T @ J_x
    :param f: A `Tensor` or a list of tensors for computing the Jacobian J_x
    :param x: A `Tensor` or a list of tensors with respect to which the Jacobian is computed.
    :param v: A `Tensor` or a list of tensors having the same shape and type as `f`
    :return: A `Tensor` or a list of tensors having the same shape and type as `x`


    got this func from https://gist.github.com/yang-song/07392ed7d57a92a87968e774aef96762
    """
    assert not isinstance(f, list) or isinstance(
        v, list
    ), "f and v should be of the same type"
    return gradients(f, x, grad_ys=v)


def setup_param_logging(all_params):
    summaries = []
    nparams = len(all_params)
    for i in range(nparams):
        param = all_params[i]
        param_shape = tuple(param.get_shape().as_list())
        for ii in range(param_shape[0]):
            if len(param_shape) == 1 or (len(param_shape) < 2 and param_shape[1] == 1):
                summaries.append(
                    tf.summary.scalar("%s_%d" % (param.name[:-2], ii + 1), param[ii])
                )
            else:
                for jj in range(param_shape[1]):
                    summaries.append(
                        f.summary.scalar(
                            "%s_%d%d" % (param.name[:-2], ii + 1, jj + 1), param[ii, jj]
                        )
                    )
    return summaries



# Gabriel's MMD stuff
def MMD2u(K, m, n):
    """The MMD^2_u unbiased statistic.
    """
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return (
        1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum())
        + 1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum())
        - 2.0 / (m * n) * Kxy.sum()
    )


def compute_null_distribution(
    K, m, n, iterations=10000, verbose=False, random_state=None, marker_interval=1000
):
    """Compute the bootstrap null-distribution of MMD2u.
    """
    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        idx = rng.permutation(m + n)
        K_i = K[idx, idx[:, None]]
        mmd2u_null[i] = MMD2u(K_i, m, n)

    if verbose:
        print("")

    return mmd2u_null


def kernel_two_sample_test(
    X,
    Y,
    kernel_function="rbf",
    iterations=10000,
    verbose=False,
    random_state=None,
    **kwargs
):
    """Compute MMD^2_u, its null distribution and the p-value of the
    kernel two-sample test.
    Note that extra parameters captured by **kwargs will be passed to
    pairwise_kernels() as kernel parameters. E.g. if
    kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1),
    then this will result in getting the kernel through
    kernel_function(metric='rbf', gamma=0.1).
    """
    m = len(X)
    n = len(Y)
    XY = np.vstack([X, Y])
    K = pairwise_kernels(XY, metric=kernel_function, **kwargs)
    mmd2u = MMD2u(K, m, n)
    if verbose:
        print("MMD^2_u = %s" % mmd2u)
        print("Computing the null distribution.")

    mmd2u_null = compute_null_distribution(
        K, m, n, iterations, verbose=verbose, random_state=random_state
    )
    p_value = max(1.0 / iterations, (mmd2u_null > mmd2u).sum() / float(iterations))
    if verbose:
        print("p-value ~= %s \t (resolution : %s)" % (p_value, 1.0 / iterations))

    return mmd2u, mmd2u_null, p_value


def rvs(dim):
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = np.eye(dim - n + 1) - 2.0 * np.outer(x, x) / (x * x).sum()
        mat = np.eye(dim)
        mat[n - 1 :, n - 1 :] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H

    def compute_K_tf(tau, T, T_s, dtype=tf.float64):
        diffs = T_s * tf.range(T, dtype=dtype)
        gammas = tf.exp(-(tf.square(diffs) / (2 * tf.square(tau))))

        gammas_rev = tf.reverse(gammas, axis=[0])
        # reverse list

        K_rows = []
        for i in range(T):
            first = gammas_rev[T - (i + 1) : (T - 1)]
            second = gammas_rev[: (T - i)]
            row_i = tf.concat((gammas_rev[T - (i + 1) : (T - 1)], gammas[: (T - i)]), 0)
            K_rows.append(row_i)

        K = tf.convert_to_tensor(K_rows)
        return K


def initialize_nf(system, arch_dict, sigma_init, random_seed):
    if (system.density_network_bounds is not None):
        a, b = system.density_network_bounds
    else:
        a = None
        b = None
    initdir = get_initdir(arch_dict,
                          random_seed,
                          init_type='gauss',
                          mu=system.mu,
                          sigma=sigma_init*np.ones((system.D)),
                          a=a,
                          b=b)
    initialized = check_init(initdir)
    if (not initialized):
        initialize_gauss_nf(system.D,
                            arch_dict,
                            sigma_init,
                            random_seed,
                            initdir,
                            mu=system.density_network_init_mu,
                            bounds=system.density_network_bounds)
    return initdir


def initialize_gauss_nf(D, arch_dict, sigma_init, random_seed, gauss_initdir, mu=None, bounds=None):
    if (bounds is not None):
        # make this more flexible for single bounds
        fam_class = family_from_str("truncated_normal")
        family = fam_class(D, a=bounds[0], b=bounds[1])
    else:
        fam_class = family_from_str("normal")
        family = fam_class(D)

    if (mu is None):
        mu = np.zeros((D,))

    params = {
        "mu": mu,
        "Sigma": np.square(sigma_init) * np.eye(D),
        "dist_seed": 0,
    }
    n = 1000
    lr_order = -3
    check_rate = 100
    min_iters = 10000
    max_iters = 10000
    converged = False
    while (not converged):
        converged = train_nf(
            family,
            params,
            arch_dict,
            n,
            lr_order,
            random_seed,
            min_iters,
            max_iters,
            check_rate,
            None,
            profile=False,
            savedir=gauss_initdir,
        )
        if converged:
            print("done initializing gaussian NF")
        else:
            max_iters = 4*max_iters
    return converged
