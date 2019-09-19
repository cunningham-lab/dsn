import tensorflow as tf
import numpy as np
import scipy
from tf_util.stat_util import approx_equal
from dsn.util.systems import (
    system,
    Linear2D,
    STGCircuit,
    V1Circuit,
    SCCircuit,
    LowRankRNN,
)
from dsn.util.dsn_util import get_system_from_template
import matplotlib.pyplot as plt

DTYPE = tf.float64
EPS = 1e-16


class sc_circuit:
    def __init__(self,):
        # time course for task
        self.t_cue_delay = 1.2
        self.t_choice = 0.6
        t_total = self.t_cue_delay + self.t_choice
        self.dt = 0.024
        self.t = np.arange(0.0, t_total, self.dt)
        self.T = self.t.shape[0]

    def simulate(self, W, Evals, w, eta):
        # declare params
        dt = 0.024
        theta = 0.05
        beta = 0.5
        tau = 0.09
        sigma = 1.0

        # make inputs
        E_constant, E_Pbias, E_Prule, E_Arule, E_choice, E_light = Evals

        I_constant = np.tile(E_constant * np.array([[1, 1, 1, 1]]), (self.T, 1))

        I_Pbias = np.zeros((self.T, 4))
        I_Pbias[self.t < 1.2] = E_Pbias * np.array([1, 0, 0, 1])

        I_Prule = np.zeros((self.T, 4))
        I_Prule[self.t < 1.2] = E_Prule * np.array([1, 0, 0, 1])

        I_Arule = np.zeros((self.T, 4))
        I_Arule[self.t < 1.2] = E_Arule * np.array([0, 1, 1, 0])

        I_choice = np.zeros((self.T, 4))
        I_choice[self.t > 1.2] = E_choice * np.array([1, 1, 1, 1])

        I_lightL = np.zeros((self.T, 4))
        I_lightL[self.t > 1.2] = E_light * np.array([1, 1, 0, 0])

        I_lightR = np.zeros((self.T, 4))
        I_lightR[self.t > 1.2] = E_light * np.array([0, 0, 1, 1])

        I_LP = I_constant + I_Pbias + I_Prule + I_choice + I_lightL
        I_LA = I_constant + I_Pbias + I_Arule + I_choice + I_lightL

        u = np.zeros((2, self.T, 4))
        v = np.zeros((2, self.T, 4))

        # initialization
        v0 = np.array([0.1, 0.1, 0.1, 0.1])
        u0 = beta * np.arctanh(2 * v0 - 1) - theta

        v[0,0] = v0
        v[1,0] = v0
        u[0,0] = u0
        u[1,0] = u0

        for i in range(1, self.T):
            du_LP = (dt / tau) * (-u[0,i - 1] + np.dot(W, v[0,i - 1]) + I_LP[i] + sigma * w[i])
            du_LA = (dt / tau) * (-u[1,i - 1] + np.dot(W, v[1,i - 1]) + I_LA[i] + sigma * w[i])
            u[0,i] = u[0,i - 1] + du_LP
            u[1,i] = u[1,i - 1] + du_LA

            for c in range(2):
                v[c,i] = eta[i] * (0.5 * np.tanh((u[c,i] - theta) / beta) + 0.5)

        return v


def test_SCCircuit():
    EPS = 1e-12
    DTYPE = tf.float64

    np.random.seed(0)
    M = 100
    sess = tf.Session()
    # true_sys = v1_circuit(T=T, dt=dt)
    Z = tf.placeholder(dtype=DTYPE, shape=(1, M, None))

    p = 0.7
    std = 0.05
    var = std**2
    inact_str = "NI"
    param_dict = {"behavior_type": "WTA", "p": p, "var": var, "inact_str": inact_str}

    system = get_system_from_template("SCCircuit", param_dict)

    print(system.behavior_str)
    assert (
        system.behavior_str
        == "WTA_mu=7.00E-01_7.00E-01_2.50E-03_2.50E-03_0.00E+00_0.00E+00_1.00E+00_1.00E+00"
    )
    assert system.D == 8
    assert system.num_suff_stats == 8

    # Test simulation
    _Z = np.random.normal(0.0, 3.0, (1, M, system.D))

    r_t = system.simulate(Z)
    _r_t = sess.run(r_t, {Z: _Z})

    true_sys = sc_circuit()

    _sW_P = _Z[0, :, 0]
    _sW_A = _Z[0, :, 1]
    _vW_PA = _Z[0, :, 2]
    _vW_AP = _Z[0, :, 3]
    _dW_PA = _Z[0, :, 4]
    _dW_AP = _Z[0, :, 5]
    _hW_P = _Z[0, :, 6]
    _hW_A = _Z[0, :, 7]

    W = np.array(
        [
            [_sW_P, _vW_PA, _dW_PA, _hW_P],
            [_vW_AP, _sW_A, _hW_A, _dW_AP],
            [_dW_AP, _hW_A, _sW_A, _vW_AP],
            [_hW_P, _dW_PA, _vW_PA, _sW_P],
        ]
    )
    W = np.transpose(W, [2, 0, 1])

    E_constant = 0.0
    E_Pbias = 0.0
    E_Prule = 10.0
    E_Arule = 10.0
    E_choice = 2.0
    E_light = 1.0
    Es = [E_constant, E_Pbias, E_Prule, E_Arule, E_choice, E_light]

    _w = system.w
    N = _w.shape[4]

    eta_NI = np.ones((system.T,))

    r_t_true = np.zeros((2, system.T, M, 4, system.N))
    for i in range(M):
        for j in range(N):
            _r_t_true_ij = true_sys.simulate(W[i], Es, _w[:, 0, 0, :, j], eta_NI)
            r_t_true[:, :, i, :, j] = _r_t_true_ij
    r_t_true = np.transpose(r_t_true, [1, 0, 2, 3, 4])
    assert approx_equal(_r_t, r_t_true, EPS)

    return None


if __name__ == "__main__":
    test_SCCircuit()
