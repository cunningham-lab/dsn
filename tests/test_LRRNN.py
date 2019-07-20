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


class low_rank_rnn:
    def __init__(self,):
        pass

    def compute_mu(self, behavior):
        means = behavior["means"]
        variances = behavior["variances"]
        first_mom = means
        second_mom = means ** 2 + variances
        return np.concatenate((first_mom, second_mom), axis=0)

    def compute_suff_stats(self, g, Mm, Mn, Sm):
        y0 = [50.0, 55.0, 45.0]
        VecPar = [Mm, Mn, Sm, 0.0]
        y = mf.SolveChaotic(y0, g, VecPar, tolerance=1e-4, backwards=1)
        mu = y[0]
        delta_0 = y[1]
        delta_inf = y[2]
        return np.array(
            [
                mu,
                delta_inf,
                delta_0 - delta_inf,
                mu ** 2,
                delta_inf ** 2,
                (delta_0 - delta_inf) ** 2,
            ]
        )
n = 100

def test_LowRankRNN():
    n = 30
    with tf.Session() as sess:
        true_sys = low_rank_rnn()
        Z = tf.placeholder(dtype=DTYPE, shape=(1, n, None))

        mu1 = np.array([1.0, 1.0, 1.0])
        Sigma1 = np.array([0.1, 0.1, 0.1])
        behavior1 = {"type": "struct_chaos", "means": mu1, "variances": Sigma1}
        model_opts = {"rank": 1, "input_type": "spont"}

        # no fixed parameters
        sys = LowRankRNN({}, behavior1, model_opts=model_opts)
        assert sys.name == "LowRankRNN"
        assert (
            sys.behavior_str
            == "struct_chaos_mu=1.00E+00_1.00E+00_1.00E+00_1.10E+00_1.10E+00_1.10E+00"
        )
        assert not sys.fixed_params  # empty dict evaluates to False
        assert sys.all_params == ["g", "Mm", "Mn", "Sm"]
        assert sys.free_params == ["g", "Mm", "Mn", "Sm"]
        assert sys.z_labels == [r"$g$", r"$M_m$", r"$M_n$", r"$\Sigma_m$"]
        assert sys.T_x_labels == [
            r"$\mu$",
            r"$\Delta_{\infty}$",
            r"$\Delta_T$",
            r"$\mu^2$",
            r"$\Delta_{\infty}^2$",
            r"$(\Delta_T)^2$",
        ]
        assert sys.D == 4
        assert sys.num_suff_stats == 6

    return None


if __name__ == "__main__":
     test_LowRankRNN()
