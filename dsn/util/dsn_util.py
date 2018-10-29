import tensorflow as tf
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_kernels
from scipy.stats import ttest_1samp, multivariate_normal
import statsmodels.sandbox.distributions.mv_normal as mvd
import matplotlib.pyplot as plt
from tf_util.flows import AffineFlowLayer, PlanarFlowLayer, SimplexBijectionLayer, \
                        CholProdLayer, StructuredSpinnerLayer, TanhLayer, ExpLayer, \
                        SoftPlusLayer, GP_EP_CondRegLayer, GP_Layer, AR_Layer, VAR_Layer, \
                        FullyConnectedFlowLayer, ElemMultLayer
from tf_util.tf_util import count_layer_params, get_flowstring
import scipy.linalg

def setup_IO(system, flow_dict, lr_order, c_init_order, random_seed, dir_str):
    # set file I/O stuff
    resdir = 'results/' + dir_str + '/';
    flowstring = get_flowstring(flow_dict);
    savedir = resdir + '%s_D=%d_T=%d_flow=%s_lr_order=%d_c=%d_rs=%d/' % \
              (system.name, system.D, system.T, flowstring, lr_order, c_init_order, random_seed);
    return savedir

def get_initdir(D, flow_dict, sigma):
    # set file I/O stuff
    initdir = 'results/inits/';
    flowstring = get_flowstring(flow_dict);
    initdir = initdir + 'D=%d_%s_sigma=%.2f/' % \
              (D, flowstring, sigma);
    return initdir

def AR_to_autocov_np(alpha, sigma_eps, T):
    p = alpha.shape[0];
    if (T-1 < p):
        p = T-1;
    if (p==1):
        gamma0 = (sigma_eps**2) / (1 - (alpha[0]**2));
        gamma1 = alpha[0]*gamma0;
        gamma = np.zeros((T,));
        gamma[:2] = np.array([gamma0, gamma1]);
        for i in range(2,T):
            gamma[i] = alpha[0]*gamma[i-1];
        
    elif (p==2):
        gamma0 = (sigma_eps**2) / (1 - ((alpha[0]**2 + alpha[1]*alpha[0]**2) / (1 - alpha[1])) - (alpha[1]**2));
        gamma1 = (alpha[0]*gamma0) / (1 - alpha[1]);
        gamma2 = alpha[0]*gamma1 + alpha[1]*gamma0;
        gamma = np.zeros((T,));
        gamma[:3] = np.array([gamma0, gamma1, gamma2]);
        for i in range(3, T):
            gamma[i] = alpha[0]*gamma[i-1] + alpha[1]*gamma[i-2];
            
    elif (p==3):
        beta = (alpha[0] + alpha[1]*alpha[2]) / ((1 - alpha[1])*(1-((alpha[2]*(alpha[0]+alpha[2])) / (1 - alpha[1]))));
        
        term1_denom = 1 - (alpha[1]**2) - (alpha[2]**2) - (alpha[0]*alpha[1]*alpha[2]);
        term1_denom = term1_denom - (alpha[0] + alpha[1]*(alpha[0]+alpha[2]) + alpha[1]*alpha[2] + (alpha[0]*alpha[2]*(alpha[0]+alpha[2])))*beta;
        
        gamma0 = (sigma_eps**2) / term1_denom;
        gamma1 = beta*gamma0;
        gamma2 = (alpha[0] + alpha[2])*gamma1 + alpha[1]*gamma0;
        gamma3 = alpha[0]*gamma2 + alpha[1]*gamma1 + alpha[2]*gamma0;
        
        gamma = np.zeros((T,));
        gamma[:4] = np.array([gamma0, gamma1, gamma2, gamma3]);
        for i in range(4, T):
            gamma[i] = alpha[0]*gamma[i-1] + alpha[1]*gamma[i-2] + alpha[2]*gamma[i-3];
        
        
    else:
        raise NotImplementedError();
        
    Sigma = scipy.linalg.toeplitz(gamma, gamma);
    return Sigma

def AR_to_autocov_tf(alpha, sigma_eps, P, T):
    if (T-1 < P):
        P = T-1;
    if (P==1):
        gamma0 = (sigma_eps**2) / (1 - (alpha[0]**2));
        gamma1 = alpha[0]*gamma0;
        gammas = [gamma0, gamma1];
        for i in range(2,T):
            gammas.append(alpha[0]*gammas[i-1]);
        
    elif (P==2):
        gamma0 = (sigma_eps**2) / (1 - ((alpha[0]**2 + alpha[1]*alpha[0]**2) / (1 - alpha[1])) - (alpha[1]**2));
        gamma1 = (alpha[0]*gamma0) / (1 - alpha[1]);
        gamma2 = alpha[0]*gamma1 + alpha[1]*gamma0;
        gammas = [gamma0, gamma1, gamma2];
        for i in range(3, T):
            gammas.append(alpha[0]*gammas[i-1] + alpha[1]*gammas[i-2]);
            
    elif (P==3):
        beta = (alpha[0] + alpha[1]*alpha[2]) / ((1 - alpha[1])*(1-((alpha[2]*(alpha[0]+alpha[2])) / (1 - alpha[1]))));
        
        term1_denom = 1 - (alpha[1]**2) - (alpha[2]**2) - (alpha[0]*alpha[1]*alpha[2]);
        term1_denom = term1_denom - (alpha[0] + alpha[1]*(alpha[0]+alpha[2]) + alpha[1]*alpha[2] + (alpha[0]*alpha[2]*(alpha[0]+alpha[2])))*beta;
        
        gamma0 = (sigma_eps**2) / term1_denom;
        gamma1 = beta*gamma0;
        gamma2 = (alpha[0] + alpha[2])*gamma1 + alpha[1]*gamma0;
        gamma3 = alpha[0]*gamma2 + alpha[1]*gamma1 + alpha[2]*gamma0;
        
        gammas = [gamma0, gamma1, gamma2, gamma3];
        for i in range(4, T):
            gammas.append(alpha[0]*gammas[i-1] + alpha[1]*gammas[i-2] + alpha[2]*gammas[i-3]);
        
        
    else:
        raise NotImplementedError();
        
    gammas_rev = gammas[::-1]; # reverse list
        
    Sigma_rows = [];
    for i in range(T):
        first = gammas_rev[T-(i+1):(T-1)];
        second = gammas_rev[:(T-i)];
        row_i = gammas_rev[T-(i+1):(T-1)] + gammas[:(T-i)];
        Sigma_rows.append(row_i);

    Sigma = tf.convert_to_tensor(Sigma_rows);
    
    return Sigma;

def construct_latent_dynamics(flow_dict, D_Z, T):
    latent_dynamics = flow_dict['latent_dynamics'];
    inits = flow_dict['inits'];
    if ('lock' in flow_dict):
        lock = flow_dict['lock'];
    else:
        lock = False;

    if (latent_dynamics == 'GP'):
        layer = GP_Layer('GP_Layer', dim=D_Z, \
                         inits=inits, lock=lock);

    elif (latent_dynamics == 'AR'):
        param_init = {'alpha_init':inits['alpha_init'], 'sigma_init':inits['sigma_init']};
        layer = AR_Layer('AR_Layer', dim=D_Z, T=T, P=flow_dict['P'], \
                      inits=inits, lock=lock);

    elif (latent_dynamics == 'VAR'):
        param_init = {'A_init':inits['A_init'], 'sigma_init':inits['sigma_init']};
        layer = VAR_Layer('VAR_Layer', dim=D_Z, T=T, P=flow_dict['P'], \
                      inits=inits, lock=lock);

    else:
        raise NotImplementedError();

    return [layer];


def construct_time_invariant_flow(flow_dict, D_Z, T):
    layer_ind = 1;
    layers = [];
    TIF_flow_type = flow_dict['TIF_flow_type'];
    repeats = flow_dict['repeats'];

    if (TIF_flow_type == 'ScalarFlowLayer'):
        flow_class = ElemMultLayer;
        name_prefix = 'ScalarFlow_Layer';

    elif (TIF_flow_type == 'FullyConnectedFlowLayer'):
        flow_class = FullyConnectedFlowLayer;
        name_prefix = FullyConnectedFlow_Layer;

    elif (TIF_flow_type == 'LinearFlowLayer'):
        flow_class = LinearFlowLayer;
        name_prefix = 'LinearFlow_Layer';

    elif (TIF_flow_type == 'StructuredSpinnerLayer'):
        flow_class = StructuredSpinnerLayer
        name_prefix = 'StructuredSpinner_Layer';

    elif (TIF_flow_type == 'PlanarFlowLayer'):
        flow_class = PlanarFlowLayer
        name_prefix = 'PlanarFlow_Layer';

    elif (TIF_flow_type == 'TanhLayer'):
        flow_class = TanhLayer;
        name_prefix = 'Tanh_Layer';

    else:
        raise NotImplementedError();

    for i in range(repeats):
        layers.append(flow_class('%s%d' % (name_prefix, layer_ind), D_Z));
        layer_ind += 1;
        
    return layers;


def declare_theta(flow_layers):
    L_flow = len(flow_layers);
    theta =[];
    for i in range(L_flow):
        layer = flow_layers[i];
        layer_name, param_names, param_dims, initializers, lock = layer.get_layer_info();
        nparams = len(param_names);
        layer_i_params = [];
        for j in range(nparams):
            if (lock):
                param_ij = initializers[j];
            else:
                if (isinstance(initializers[j], tf.Tensor)):
                    print(initializers[j]);
                    param_ij = tf.get_variable(layer_name+'_'+param_names[j], \
                                               dtype=tf.float64, \
                                               initializer=initializers[j]);
                else:
                    print(initializers[j]);
                    param_ij = tf.get_variable(layer_name+'_'+param_names[j], shape=param_dims[j], \
                                               dtype=tf.float64, \
                                               initializer=initializers[j]);
            layer_i_params.append(param_ij);
        theta.append(layer_i_params);
    return theta;

def connect_flow(Z, layers, theta, ts=None):
    Z_shape = tf.shape(Z);
    K = Z_shape[0];
    M = Z_shape[1];
    D_Z = Z_shape[2];
    T = Z_shape[3];

    sum_log_det_jacobians = tf.zeros((K,M), dtype=tf.float64);
    nlayers = len(layers);
    Z_by_layer = [];
    Z_by_layer.append(Z);
    print('zshapes in');
    print('connect flow');
    for i in range(nlayers):
        print(Z.shape);
        layer = layers[i];
        print(i, layer.name);
        theta_layer = theta[i];
        layer.connect_parameter_network(theta_layer);
        if (isinstance(layer, GP_Layer) or isinstance(layer, GP_EP_CondRegLayer)):
            Z, sum_log_det_jacobians = layer.forward_and_jacobian(Z, sum_log_det_jacobians, ts);
        else:
            Z, sum_log_det_jacobians = layer.forward_and_jacobian(Z, sum_log_det_jacobians);
        Z_by_layer.append(Z);
    print(Z.shape);
    return Z, sum_log_det_jacobians, Z_by_layer;

p_eps = 10e-6;
def initialize_optimization_parameters(sess, optimizer, all_params):
    nparams = len(all_params);
    slot_names = optimizer.get_slot_names();
    num_slot_names = len(slot_names);
    for i in range(nparams):
        param = all_params[i];
        for j in range(num_slot_names):
            slot_name = slot_names[j];
            opt_var = optimizer.get_slot(param, slot_name);
            sess.run(opt_var.initializer);
    # Adam optimizer is the only tf optimizer that has this slot variable issue
    # href https://github.com/tensorflow/tensorflow/issues/8057
    if (isinstance(optimizer, tf.contrib.optimizer_v2.AdamOptimizer)):
        beta1_power, beta2_power = optimizer._get_beta_accumulators();
        sess.run(beta1_power.initializer);
        sess.run(beta2_power.initializer);
    return None;

def check_convergence(to_check, cur_ind, lag, thresh, criteria='lag_diff', wsize=50):
    len_to_check = len(to_check);    
    vals = to_check[0];
    for i in range(1,len_to_check):
        vals = np.concatenate((vals, to_check[1]), axis=1);

    if (criteria=='lag_diff'):
        lag_mean = np.mean(vals[(cur_ind-(lag+wsize)):(cur_ind-lag),:], axis=0);
        cur_mean = np.mean(vals[(cur_ind-wsize):cur_ind,:], axis=0);
        log_param_diff = np.log(np.linalg.norm(lag_mean-cur_mean));
        has_converged = log_param_diff < thresh;
    elif (criteria=='grad_mean_ttest'):
        last_grads = vals[(cur_ind-lag):cur_ind, :];
        Sigma_grads = np.dot(last_grads.T, last_grads) / (lag); # zero-mean covariance
        nvars = last_grads.shape[1];
        #mvt = mvd.MVT(np.zeros((nvars,)), Sigma_grads, lag);
        #grad_mean = np.mean(last_grads, 0);
        #t_cdf = mvt.cdf(grad_mean);
        #has_converged = (t_cdf > (thresh/2) and t_cdf < (1-(thresh/2)));
        #print('cdf val', t_cdf, 'convergence', has_converged);
        has_converged = True;
        for i in range(nvars):
            t, p = ttest_1samp(last_grads[:,i], 0);
            # if any grad mean is not zero, reject
            if (p < thresh):
                has_converged = False;
                break;
    return has_converged;

def approxKL(y_k, X_k, constraint_type, params, plot=False):
    log_Q = y_k[:,0];
    Q = np.exp(log_Q);
    if (constraint_type == 'normal'):
        mu = params['mu'];
        Sigma = params['Sigma'];
        dist = multivariate_normal(mean=mu, cov=Sigma);
        log_P = dist.logpdf(X_k);
        if (plot):
            n = X_k.shape[0];
            sizes = 40*np.ones((n,1));
            minval = min(np.min(log_Q), np.min(log_P));
            maxval = max(np.max(log_Q), np.max(log_P));
            fig = plt.figure(figsize=(12,4));
            fig.add_subplot(1,3,1);
            plt.scatter(X_k[:,0], X_k[:,1], sizes, log_Q, vmin=minval, vmax=maxval);
            plt.colorbar();
            fig.add_subplot(1,3,2);
            plt.scatter(X_k[:,0], X_k[:,1], sizes, log_P);
            plt.colorbar();
            fig.add_subplot(1,3,3);
            plt.scatter(X_k[:,0], X_k[:,1], sizes, log_Q-log_P);
            plt.colorbar();
            plt.show();

    Elogdiffs = log_Q - log_P;
    KL = np.sum(np.multiply(Q, log_Q - log_P));
    return KL;

    


def computeMoments(X, constraint_id):
    X_shape = tf.shape(X);
    batch_size = X_shape[0];
    D_Z, K_eta, params, constraint_type = load_constraint_info(constraint_id);
    T = X_shape[2];
    if (constraint_type == 'normal'):
        D_X = D_Z;
        cov_con_mask = np.triu(np.ones((D_X,D_X), dtype=np.bool_), 0);
        X_flat = tf.reshape(tf.transpose(X, [0, 2, 1]), [T*batch_size, D_X]); # samps x D
        Tx_mean = X_flat;
        XXT = tf.matmul(tf.expand_dims(X_flat, 2), tf.expand_dims(X_flat, 1));

        #Tx_cov = tf.transpose(tf.boolean_mask(tf.transpose(X_cov, [1,2,0]), _cov_con_mask)); # [n x (D*(D-1)/2 )]
        Tx_cov = tf.reshape(XXT, [T*batch_size, D_X*D_X]);
        Tx = tf.concat((Tx_mean, Tx_cov), axis=1);
    elif (constraint_type == 'dirichlet'):
        D_X = D_Z + 1;
        X_flat = tf.reshape(tf.transpose(X, [0, 2, 1]), [T*batch_size, D_X]); # samps x D
        Tx_log = tf.log(X_flat);
        Tx = Tx_log;
    else:
        raise NotImplementedError;


    return Tx;

def getEtas(constraint_id, K_eta):
    print(K_eta);
    D_Z, K_eta, params, constraint_type = load_constraint_info(constraint_id);
    print(K_eta);
    datadir = 'constraints/'
    fname = datadir + '%s.npz' % constraint_id;
    confile = np.load(fname);
    if (constraint_type == 'normal'):
        mu_targs = confile['mu_targs'];
        Sigma_targs = confile['Sigma_targs'];
        etas = [];
        for k in range(K_eta):
            print(k, Sigma_targs[k,:,:]);
            eta1 = np.float64(np.dot(np.linalg.inv(Sigma_targs[k,:,:]), np.expand_dims(mu_targs[k,:], 2)));
            eta2 = np.float64(-np.linalg.inv(Sigma_targs[k,:,:]) / 2);
            eta = np.concatenate((eta1, np.reshape(eta2, [D_Z*D_Z, 1])), 0);
            etas.append(eta);

        mu_OL_targs = confile['mu_OL_targs'];
        Sigma_OL_targs = confile['Sigma_OL_targs'];
        off_lattice_etas = [];
        for k in range(K_eta):
            ol_eta1 = np.float64(np.dot(np.linalg.inv(Sigma_OL_targs[k,:,:]), np.expand_dims(mu_OL_targs[k,:], 2)));
            ol_eta2 = np.float64(-np.linalg.inv(Sigma_OL_targs[k,:,:]) / 2);
            ol_eta = np.concatenate((eta1, np.reshape(eta2, [D_Z*D_Z, 1])), 0);
            off_lattice_etas.append(ol_eta);


    elif (constraint_type == 'dirichlet'):
        alpha_targs = np.float64(confile['alpha_targs']);
        etas = [];
        for k in range(K_eta):
            eta = np.expand_dims(alpha_targs[k,:], 1);
            etas.append(eta);

        alpha_OL_targs = np.float64(confile['alpha_OL_targs']);
        off_lattice_etas = [];
        for k in range(K_eta):
            ol_eta = np.expand_dims(alpha_OL_targs[k,:], 1);
            off_lattice_etas.append(ol_eta);
    return etas, off_lattice_etas;

def autocovariance(X, tau_max, T, batch_size):
    # need to finish this
    X_toep = [];
    X_toep1 = [];
    X_toep2 = [];
    for i in range(tau_max+1):
        X_toep.append(X[:,:,i:((T-tau_max)+i)]);  # This will be (n x D x tau_max x (T- tau_max))
        X_toep1.append(X[:(batch_size//2),:,i:((T-tau_max)+i)]);
        X_toep2.append(X[(batch_size//2):,:,i:((T-tau_max)+i)]);

    X_toep = tf.reshape(tf.transpose(tf.convert_to_tensor(X_toep), [2, 0, 3, 1]), [D, tau_max+1, batch_size*(T-tau_max)]); # D x tau_max x (T-tau_max)*n
    X_toep1 = tf.reshape(tf.transpose(tf.convert_to_tensor(X_toep1), [2, 0, 3, 1]), [D, tau_max+1, (batch_size//2)*(T-tau_max)]); # D x tau_max x (T-tau_max)*n
    X_toep2 = tf.reshape(tf.transpose(tf.convert_to_tensor(X_toep2), [2, 0, 3, 1]), [D, tau_max+1, (batch_size//2)*(T-tau_max)]); # D x tau_max x (T-tau_max)*n
    
    X_toep_mc = X_toep - tf.expand_dims(tf.reduce_mean(X_toep, 2), 2);
    X_toep_mc1 = X_toep1 - tf.expand_dims(tf.reduce_mean(X_toep1, 2), 2);
    X_toep_mc2 = X_toep2 - tf.expand_dims(tf.reduce_mean(X_toep2, 2), 2);

    X_tau =  tf.cast((1/(batch_size*(T-tau_max))), tf.float64)*tf.matmul(X_toep_mc[:,:,:], tf.transpose(X_toep_mc, [0,2,1]))[:,:,0];
    X_tau1 =  tf.cast((1/((batch_size//2)*(T-tau_max))), tf.float64)*tf.matmul(X_toep_mc1[:,:,:], tf.transpose(X_toep_mc1, [0,2,1]))[:,:,0];
    X_tau2 =  tf.cast((1/((batch_size//2)*(T-tau_max))), tf.float64)*tf.matmul(X_toep_mc2[:,:,:], tf.transpose(X_toep_mc2, [0,2,1]))[:,:,0];

    X_tau_err = tf.reshape(X_tau - autocov_targ[:,:(tau_max+1)], (D*(tau_max+1),));
    X_tau_err1 = tf.reshape(X_tau1 - autocov_targ[:,:(tau_max+1)], (D*(tau_max+1),));
    X_tau_err2 = tf.reshape(X_tau2 - autocov_targ[:,:(tau_max+1)], (D*(tau_max+1),));
    tau_MSE = tf.reduce_sum(tf.square(X_tau_err));
    Tx_autocov = 0;
    Rx_autocov = 0;
    return Tx_autocov, Rx_autocov;


def time_invariant_flow(Z_AR, eta, layers, constraint_type):
    Z_AR_shape = tf.shape(Z_AR);
    batch_size = Z_AR_shape[0];
    D = Z_AR_shape[1];
    T = Z_AR_shape[2];
    Z_AR_flat = tf.reshape(tf.transpose(Z_AR, [0, 2, 1]), [T*batch_size, D]);

    Z = tf.transpose(Z_AR_flat);
    eta = tf.transpose(eta);
    #sum_log_det_jacobian = tf.zeros((1,T*batch_size));
    sum_log_det_jacobians = 0.0;
    for layer in layers:
        Z, sum_log_det_jacobians = layer.forward_and_jacobian(Z, eta, sum_log_det_jacobians, reuse=False);

    # final layer translates to the support
    if (constraint_type == 'dirichlet'):
        Z = tf.exp(Z) / (tf.reduce_sum(tf.exp(Z) ,axis=0) + 1); 
        # compute the jacobian using matrix determinant lemma
        u = Z;
        Adiag = u;
        Ainvdiag = 1.0 / u;
        v = -u;
        g_det_jacobian = (1.0+tf.reduce_sum(tf.multiply(tf.multiply(v,Ainvdiag), u), axis=0))*tf.reduce_prod(Adiag, axis=0);
        g_log_det_jacobian = tf.log(g_det_jacobian);
        sum_log_det_jacobians += g_log_det_jacobian;
        Z = tf.concat((Z, tf.expand_dims(1-tf.reduce_sum(Z, axis=0), 0)), axis=0);

    return tf.transpose(Z), tf.transpose(sum_log_det_jacobians);


def compute_R2(log_q_x, log_h_x, T_x_in):
    T_x_shape = tf.shape(T_x_in);
    M = T_x_shape[1];
    num_suff_stats = T_x_shape[2];

    T_x_in = T_x_in[0] - tf.expand_dims(tf.reduce_mean(T_x_in[0], 0), 0);
    T_x = tf.concat((T_x_in, tf.ones((M, 1), tf.float64)), axis=1);
    prec_mat = tf.matmul(tf.transpose(T_x), T_x); # + c*tf.eye(num_suff_stats+1, dtype=tf.float64);

    if (log_h_x is not None):
        y_q = tf.expand_dims(log_q_x[0] - log_h_x[0], 1);
    else:
        y_q = tf.expand_dims(log_q_x[0], 1);
        
    beta_q = tf.matmul(tf.matrix_inverse(prec_mat), tf.matmul(tf.transpose(T_x), y_q));
    
    # compute optimial linear regression offset term for eta
    residuals_q = y_q - tf.matmul(T_x, beta_q);
    RSS = tf.matmul(tf.transpose(residuals_q), residuals_q);
    y_q_mc = y_q - tf.reduce_mean(y_q);
    TSS = tf.matmul(tf.transpose(y_q_mc), y_q_mc)
    # compute the R^2 of the exponential family fit
    R2_q = 1.0 - (RSS[0,0] / TSS[0,0]);
    return R2_q;

def gradients(f, x, grad_ys=None):
    '''
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
    '''

    if isinstance(x, list):
        grad = tf.gradients(f, x, grad_ys=grad_ys)
        for i in range(len(x)):
            if grad[i] is None:
                grad[i] = tf.zeros_like(x[i])
        return grad
    else:
        grad = tf.gradients(f, x, grad_ys=grad_ys)[0]
        if grad is None:
            return tf.zeros_like(x)
        else:
            return grad

def Lop(f, x, v):
    '''
    Compute Jacobian-vector product. The result is v^T @ J_x
    :param f: A `Tensor` or a list of tensors for computing the Jacobian J_x
    :param x: A `Tensor` or a list of tensors with respect to which the Jacobian is computed.
    :param v: A `Tensor` or a list of tensors having the same shape and type as `f`
    :return: A `Tensor` or a list of tensors having the same shape and type as `x`


    got this func from https://gist.github.com/yang-song/07392ed7d57a92a87968e774aef96762
    '''
    assert not isinstance(f, list) or isinstance(v, list), "f and v should be of the same type"
    return gradients(f, x, grad_ys=v)

def setup_param_logging(all_params):
    summaries = [];
    nparams = len(all_params);
    for i in range(nparams):
        param = all_params[i];
        param_shape = tuple(param.get_shape().as_list());
        for ii in range(param_shape[0]):
            if (len(param_shape)==1 or (len(param_shape) < 2 and param_shape[1]==1)):
                summaries.append(tf.summary.scalar('%s_%d' % (param.name[:-2], ii+1), param[ii]));
            else:
                for jj in range(param_shape[1]):
                    summaries.append(f.summary.scalar('%s_%d%d' % (param.name[:-2], ii+1, jj+1), param[ii, jj]));
    return summaries;

def log_grads(cost_grads, cost_grad_vals, ind):
    cgv_ind = 0;
    nparams = len(cost_grads);
    for i in range(nparams):
        grad = cost_grads[i];
        grad_shape = grad.shape;
        ngrad_vals = np.prod(grad_shape);
        grad_reshape = np.reshape(grad, (ngrad_vals,));
        for ii in range(ngrad_vals):
            cost_grad_vals[ind, cgv_ind] = grad_reshape[ii];
            cgv_ind += 1;
    return None;


def adam_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999):
    ''' Adam optimizer '''
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads
    t = tf.Variable(1., 'adam_t')
    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
        if mom1>0:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
            v_t = mom1*v + (1. - mom1)*g
            v_hat = v_t / (1. - tf.pow(mom1,t))
            updates.append(v.assign(v_t))
        else:
            v_hat = g
        mg_t = mom2*mg + (1. - mom2)*tf.square(g)
        mg_hat = mg_t / (1. - tf.pow(mom2,t))
        g_t = v_hat / tf.sqrt(mg_hat + 1e-8)
        p_t = p - lr * g_t
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    updates.append(t.assign_add(1))
    return tf.group(*updates)


def compute_VAR_cov_tf(As, Sigma_eps, D, K, T):
    # initialize the covariance matrix
    zcov = [[tf.eye(D, dtype=tf.float64)]];

    # compute the lagged noise-state covariance
    gamma = [Sigma_eps];
    for t in range(1,T):
        gamma_t = tf.zeros((D,D), dtype=tf.float64);
        for k in range(1, min(t,K)+1):
            gamma_t += tf.matmul(As[k-1], gamma[t-k]);
        gamma.append(gamma_t);

    # compute the off-block-diagonal covariance blocks
    # first row
    for s in range(1, T):
        zcov_0s = tf.zeros((D,D), dtype=tf.float64);
        for k in range(1, min(s, K)+1):
            tau = s-k;
            zcov_0tau = zcov[0][tau];
            zcov_0s += tf.matmul(zcov_0tau, tf.transpose(As[k-1]));
        zcov[0].append(zcov_0s);
        zcov.append([tf.transpose(zcov_0s)]);

    # remaining rows
    for t in range(1,T):
        for s in range(t, T):
            zcov_ts = tf.zeros((D,D), dtype=tf.float64);
            # compute the contribution of lagged state-state covariances
            for k_t in range(1, min(t,K)+1):
                tau_t = t-k_t;
                for k_s in range(1, min(s,K)+1):
                    tau_s = s-k_s;
                    zcov_tauttaus = zcov[tau_t][tau_s];
                    zcov_ts += tf.matmul(As[k_t-1], tf.matmul(zcov_tauttaus, tf.transpose(As[k_s-1])));
            # compute the contribution of lagged noise-state covariances
            if (t==s):
                zcov_ts += Sigma_eps;
            for k in range(1, min(s,K)+1):
                tau_s = s-k;
                if (tau_s >= t):
                    zcov_ts += tf.matmul(tf.transpose(gamma[tau_s-t]), tf.transpose(As[k-1]));

            zcov[t].append(zcov_ts);
            if (t != s):
                zcov[s].append(tf.transpose(zcov_ts));
                
    zcov = tf.convert_to_tensor(zcov);
    Zcov = tf.reshape(tf.transpose(zcov, [0,2,1,3]), (D*T, D*T));
    return Zcov;

def compute_VAR_cov_np(As, Sigma_eps, D, K, T):
    # Compute the analytic covariance of the VAR model

    # initialize the covariance matrix
    zcov = np.zeros((D*T, D*T));

    # compute the block-diagonal covariance
    zcov[:D, :D] = np.eye(D);

    # compute the lagged noise-state covariance
    gamma = [Sigma_eps];
    for t in range(1,T):
        gamma_t = np.zeros((D,D));
        for k in range(1, min(t,K)+1):
            gamma_t += np.dot(As[k-1], gamma[t-k]);
        gamma.append(gamma_t);

    # compute the off-block-diagonal covariance blocks
    # first row
    for s in range(1, T):
        zcov_0s = np.zeros((D,D));
        for k in range(1, min(s, K)+1):
            tau = s-k;
            zcov_0tau = zcov[:D, (D*tau):(D*(tau+1))]
            zcov_0s += np.dot(zcov_0tau, As[k-1].T);
        zcov[:D, (D*s):(D*(s+1))] = zcov_0s;
        zcov[(D*s):(D*(s+1)), :D] = zcov_0s.T;
            
    # remaining rows
    for t in range(1,T):
        for s in range(t, T):
            zcov_ts = np.zeros((D,D));
            # compute the contribution of lagged state-state covariances
            for k_t in range(1, min(t,K)+1):
                tau_t = t-k_t;
                for k_s in range(1, min(s,K)+1):
                    tau_s = s-k_s;
                    zcov_tauttaus = zcov[(D*tau_t):(D*(tau_t+1)), (D*tau_s):(D*(tau_s+1))]
                    zcov_ts += np.dot(As[k_t-1], np.dot(zcov_tauttaus, As[k_s-1].T));
                
            # compute the contribution of lagged noise-state covariances
            if (s==t):
                zcov_ts += Sigma_eps;
            for k in range(1, min(s,K)+1):
                tau_s = s-k;
                if (tau_s >= t):
                    zcov_ts += np.dot(gamma[tau_s-t].T, As[k-1].T);
            
            zcov[(D*t):(D*(t+1)), (D*s):(D*(s+1))] = zcov_ts;
            zcov[(D*s):(D*(s+1)), (D*t):(D*(t+1))] = zcov_ts.T;
    return zcov;


def simulate_VAR(As, Sigma_eps, T):
    K = As.shape[0];
    D = As.shape[1];
    mu = np.zeros((D,));
    z = np.zeros((D,T));
    z[:,0] = np.random.multivariate_normal(mu, np.eye(D));
    for t in range(1, T):
        Z_VAR_pred_t = np.zeros((D,));
        for k in range(K):
            if (t-(k+1) >= 0):
                Z_VAR_pred_t += np.dot(As[k], z[:, t-(k+1)]);
        eps_t = np.random.multivariate_normal(mu, Sigma_eps);
        z[:,t] = Z_VAR_pred_t + eps_t;
    return z;

# Gabriel's MMD stuff
def MMD2u(K, m, n):
    """The MMD^2_u unbiased statistic.
    """
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
        1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
        2.0 / (m * n) * Kxy.sum()
        
def compute_null_distribution(K, m, n, iterations=10000, verbose=False,
                              random_state=None, marker_interval=1000):
    """Compute the bootstrap null-distribution of MMD2u.
    """
    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        idx = rng.permutation(m+n)
        K_i = K[idx, idx[:, None]]
        mmd2u_null[i] = MMD2u(K_i, m, n)

    if verbose:
        print("")

    return mmd2u_null

def kernel_two_sample_test(X, Y, kernel_function='rbf', iterations=10000,
                           verbose=False, random_state=None, **kwargs):
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

    mmd2u_null = compute_null_distribution(K, m, n, iterations,
                                           verbose=verbose,
                                           random_state=random_state)
    p_value = max(1.0/iterations, (mmd2u_null > mmd2u).sum() /
                  float(iterations))
    if verbose:
        print("p-value ~= %s \t (resolution : %s)" % (p_value, 1.0/iterations))

    return mmd2u, mmd2u_null, p_value

def rvs(dim):
  random_state = np.random
  H = np.eye(dim)
  D = np.ones((dim,))
  for n in range(1, dim):
   x = random_state.normal(size=(dim-n+1,))
   D[n-1] = np.sign(x[0])
   x[0] -= D[n-1]*np.sqrt((x*x).sum())
   # Householder transformation
   Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
   mat = np.eye(dim)
   mat[n-1:, n-1:] = Hx
   H = np.dot(H, mat)
   # Fix the last sign such that the determinant is 1
  D[-1] = (-1)**(1-(dim % 2))*D.prod()
  # Equivalent to np.dot(np.diag(D), H) but faster, apparently
  H = (D*H.T).T
  return H

  def compute_K_tf(tau, T, T_s, dtype=tf.float64):
    diffs = T_s*tf.range(T, dtype=dtype);
    gammas = tf.exp(-(tf.square(diffs) / (2*tf.square(tau))));

                
    gammas_rev = tf.reverse(gammas, axis=[0]); # reverse list
        
    K_rows = [];
    for i in range(T):
        first = gammas_rev[T-(i+1):(T-1)];
        second = gammas_rev[:(T-i)];
        row_i = tf.concat((gammas_rev[T-(i+1):(T-1)], gammas[:(T-i)]), 0);
        K_rows.append(row_i);

    K = tf.convert_to_tensor(K_rows);
    return K; 

