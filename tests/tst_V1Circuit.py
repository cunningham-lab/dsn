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
import matplotlib.pyplot as plt

# import dsn.lib.LowRank.Fig1_Spontaneous.fct_mf as mf

DTYPE = tf.float64
EPS = 1e-16


class v1_circuit:
    def __init__(
        self, T=40, dt=0.25, init_conds=np.array([[1.0], [1.1], [1.2], [1.3]])
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
                if drdt[i] > 1e6:
                    drdt[i] = 1e6
                if drdt[i] < -1e6:
                    drdt[i] = -1e6

            return drdt

        t = np.arange(0, self.T * self.dt, self.dt)
        y = scipy.integrate.odeint(f, r0, t)

        return y


def test_V1Circuit():
    np.random.seed(0)
    M = 100
    sess = tf.Session()
    dt = 0.05
    T = 50
    true_sys = v1_circuit(T=T, dt=dt)
    Z = tf.placeholder(dtype=DTYPE, shape=(1, M, None))

    fixed_params = {'h_FFE':0.0, \
                    'h_FFP':0.0, \
                    'h_LATE':0.0, \
                    'h_LATP':0.0, \
                    'h_LATS':0.0, \
                    'h_LATV':0.0, \
                    'n':2.0, \
                    's_0':30}

    return None


if __name__ == "__main__":
    test_V1Circuit()
