import tensorflow as tf
import numpy as np
import scipy
import scipy.io as sio
from tf_util.stat_util import approx_equal
from dsn.util.dsn_util import get_system_from_template
import matplotlib.pyplot as plt
import os

# import dsn.lib.LowRank.Fig1_Spontaneous.fct_mf as mf

DTYPE = tf.float64
EPS = 1e-16
MAX_DRDT = 1e6


class v1_circuit:
    def __init__(
        self, T=100, dt=0.005, init_conds=np.random.normal(1.0, 0.01, (4,1))
    ):
        self.init_conds = init_conds
        self.dt = dt
        self.T = T

    def compute_h(
        self,
        c,
        s,
        r,
        b,
        h_FF,
        h_LAT,
        h_RUN,
        g_FF_str,
        g_LAT_str,
        g_RUN_str,
        s_0,
        a=None,
        c_50=None,
    ):
        # Set the gates
        if g_FF_str == "c":
            g_FF = c
        elif g_FF_str == "saturate":
            assert a is not None
            assert c_50 is not None
            g_FF = (c ** a) / (c_50 ** a + c ** a)
        else:
            raise NotImplementedError()

        if g_LAT_str == "linear":
            s_diff = s_0 - s
        elif g_LAT_str == "square":
            s_diff = s_0 ** 2 - s ** 2
        else:
            raise NotImplementedError()
        if s_diff < 0:
            g_LAT = 0.0
        else:
            g_LAT = c * s_diff

        if g_RUN_str == "r":
            g_RUN = r

        h = b + g_FF * h_FF + g_LAT * h_LAT + g_RUN * h_RUN
        return h

    def simulate(self, W, h, tau, n):
        r0 = self.init_conds[:, 0]

        def f(r, t):
            pow_arg = np.dot(W, np.expand_dims(r, 1)) + h

            for i in range(4):
                if pow_arg[i, 0] < 0:
                    pow_arg[i, 0] = 0.0

            drdt = (1.0 / tau) * (-r + np.power(pow_arg[:, 0], n))

            for i in range(4):
                if drdt[i] > MAX_DRDT:
                    drdt[i] = MAX_DRDT
                if drdt[i] < -MAX_DRDT:
                    drdt[i] = -MAX_DRDT

            return drdt

        t = np.arange(0, self.T * self.dt, self.dt)
        y = scipy.integrate.odeint(f, r0, t)

        return y

def read_V1_responses(s):
    s_type = type(s)
    if (s_type == list):
        s=np.array(s)
    elif (s_type == np.ndarray):
        s=s
    elif (s_type == int):
        s=np.array([s])
    cell_ord = [3,2,0,1]
    D = len(cell_ord)
    C = len(s)
    fname = "data/V1/ProcessedData.mat"
    matfile = sio.loadmat(fname)
    s_data =  matfile["ProcessedData"]["StimulusSize_deg"][0, 0][0]
    MeanResponse = matfile["ProcessedData"]["MeanResponse"][0, 0][cell_ord,:,0]
    SEMMeanResponse = matfile["ProcessedData"]["SEMMeanResponse"][0, 0][cell_ord,:,0]
    s_inds = [np.where(s_data == i)[0][0] for i in s]
    means = np.reshape(MeanResponse[:,s_inds].T, (C*D,))
    stds = np.reshape(SEMMeanResponse[:,s_inds].T, (C*D,))
    return means, stds

def read_W(source):
    fname = "data/V1/V1_Zs.npz"
    npzfile = np.load(fname)
    W = npzfile['Z_%s_square' % source]
    return W


def test_V1Circuit():
    os.chdir('../dsn/')
    np.random.seed(0)
    M = 100
    sysname = 'V1Circuit'
    T = 100
    dt = 0.005
    W_allen = read_W('allen')

    # Test rates behavior
    #TODO at some point this should update to incorporate s into the input function
    behavior_type = 'rates'
    fac = 1.0
    s = [5]
    C = len(s)
    param_dict = {'behavior_type':behavior_type,'fac':fac,'s':s}
    system = get_system_from_template(sysname, param_dict)
    assert(len(system.z_labels) == system.D)

    # Test parameter setting
    Z = tf.placeholder(tf.float64, (1,None,system.D))
    W, b, h_FF, h_LAT, h_RUN, tau, n, s_0, a, c_50 = system.filter_Z(Z)

    _Z = np.zeros((1,M,system.D))
    Z_a,Z_b = system.density_network_bounds
    for i in range(system.D):
        _Z[0,:,i] = np.random.uniform(Z_a[i], Z_b[i], (M,))

    tf_vars = [W, b, h_FF, h_LAT, h_RUN, tau, n]
    with tf.Session() as sess:
        _tf_vars = sess.run(tf_vars, {Z:_Z})
    _W, _b, _h_FF, _h_LAT, _h_RUN, _tau, _n = _tf_vars
    for i in range(M):
        assert(approx_equal(np.abs(_W[0,i,:,:]), W_allen, EPS))
        assert(approx_equal(_b[0,i,:,0], _Z[0,i,:4], EPS))
        assert(approx_equal(_n[0,i,0], _Z[0,i,4], EPS))
    assert(approx_equal(_h_FF, 0.0, EPS))
    assert(approx_equal(_h_LAT, 0.0, EPS))
    assert(approx_equal(_h_RUN, 0.0, EPS))
    assert(approx_equal(_tau, 0.02, EPS))

    # Test input calculation
    h = system.compute_h(b, h_FF, h_LAT, h_RUN, s_0)
    with tf.Session() as sess:
        _h = sess.run(h, {Z:_Z})
    assert(approx_equal(_h[0,:,:,0], _Z[0,:,:4], EPS))

    # Test suff_stat calculation
    r_t = system.simulate(Z)
    T_x = system.compute_suff_stats(Z)
    with tf.Session() as sess:
        _r_t = sess.run(r_t, {Z:_Z})
        _T_x = sess.run(T_x, {Z:_Z})
    v1_circuit_true = v1_circuit(T=T, dt=dt, init_conds=system.init_conds)
    r_ss = np.zeros((M,4))
    for i in range(M):
        r_t_i = v1_circuit_true.simulate(_W[0,i,:,:], _h[0,i,:,:], _tau[0,i,0,0], _n[0,i,0,0]) 
        r_ss[i,:] = r_t_i[-1,:]
        #assert(approx_equal(_r_t[:,0,i,:,0], r_t_i, 1e-6)) # should make exact 
    mean_true = r_ss
    var_true = np.square(r_ss - np.expand_dims(system.mu[:4], 0))
    T_x_true = np.concatenate((mean_true, var_true), axis=1)
    assert(approx_equal(_T_x, np.expand_dims(T_x_true, 0), 1e-6))

    # Check that mu matches data
    s_list = [5, [5], [10], np.array([60])]
    fac_list = [1.0, 0.5, 100.0]
    num_s = len(s_list)
    num_fac = len(fac_list)

    for i in range(num_s):
        s = s_list[i]
        means, stds = read_V1_responses(s)
        for j in range(num_fac):
            fac = fac_list[j]
            means_true = fac*means
            stds_true = fac*stds
            vars_true = np.square(stds_true)
            print(means_true.shape, vars_true.shape)
            mu_true = np.concatenate((means_true, vars_true), axis=0)
            param_dict = {'behavior_type':behavior_type,'fac':fac,'s':s}
            system = get_system_from_template(sysname, param_dict)
            assert(len(system.T_x_labels) == system.num_suff_stats)
            assert(system.mu.shape[0] == system.num_suff_stats)
            approx_equal(mu_true, system.mu, EPS)



    # Test differences behavior
    #TODO at some point this should update to incorporate s into the input function
    behavior_type = 'difference'
    alphas = ['E', 'P', 'S', 'V']
    num_alphas = len(alphas)
    for k in range(num_alphas):
        alpha = alphas[k]
        param_dict = {
            "behavior_type":behavior_type,
            "alpha":alpha,
            "inc_val":0.5,
        }
        system = get_system_from_template(sysname, param_dict)
        assert(len(system.z_labels) == system.D)

        # Test parameter setting
        Z = tf.placeholder(tf.float64, (1,None,system.D))
        W, b, h_FF, h_LAT, h_RUN, tau, n, s_0, a, c_50 = system.filter_Z(Z)

        _Z = np.zeros((1,M,system.D))
        Z_a,Z_b = system.density_network_bounds
        for i in range(system.D):
            _Z[0,:,i] = np.random.uniform(Z_a[i], Z_b[i], (M,))

        tf_vars = [W, b, h_FF, h_LAT, h_RUN, tau, n]
        with tf.Session() as sess:
            _tf_vars = sess.run(tf_vars, {Z:_Z})
        _W, _b, _h_FF, _h_LAT, _h_RUN, _tau, _n = _tf_vars
        for i in range(M):
            assert(approx_equal(np.abs(_W[0,i,:,:]), W_allen, EPS))
            assert(approx_equal(_h_RUN[0,i,:,0], _Z[0,i,:], EPS))
        assert(approx_equal(_n, 2.0, EPS))
        assert(approx_equal(_h_FF, 0.0, EPS))
        assert(approx_equal(_h_LAT, 0.0, EPS))
        assert(approx_equal(_b, 1.0, EPS))
        assert(approx_equal(_tau, 0.02, EPS))

        # Test input calculation
        h = system.compute_h(b, h_FF, h_LAT, h_RUN, s_0)
        with tf.Session() as sess:
            _h = sess.run(h, {Z:_Z})

    # Test suff_stat calculation
        r_t = system.simulate(Z)
        T_x = system.compute_suff_stats(Z)
        with tf.Session() as sess:
            _r_t = sess.run(r_t, {Z:_Z})
            _T_x = sess.run(T_x, {Z:_Z})


        v1_circuit_true = v1_circuit(T=T, dt=dt, init_conds=system.init_conds)
        r_ss = np.zeros((M,2))
        for i in range(M):
            r_t_i_NORUN = v1_circuit_true.simulate(_W[0,i,:,:], _h[0,i,:,:], _tau[0,i,0,0], _n[0,i,0,0]) 
            r_t_i_RUN = v1_circuit_true.simulate(_W[0,i,:,:], _h[1,i,:,:], _tau[0,i,0,0], _n[0,i,0,0]) 
            r_ss[i,0] = r_t_i_NORUN[-1,k]
            r_ss[i,1] = r_t_i_RUN[-1,k]
            #assert(approx_equal(_r_t[:,0,i,:,0], r_t_i, 1e-6)) # should make exact 
        dr_ss = r_ss[:,1] - r_ss[:,0]
        mean_true = np.expand_dims(dr_ss, 1)
        var_true = np.square(mean_true - system.mu[0])
        T_x_true = np.concatenate((mean_true, var_true), axis=1)
        assert(approx_equal(_T_x, np.expand_dims(T_x_true, 0), 1e-6))


    return None


if __name__ == "__main__":
    test_V1Circuit()
