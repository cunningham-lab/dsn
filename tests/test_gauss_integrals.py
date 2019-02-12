import dsn.util.tf_integrals as tfi
import dsn.util.fct_integrals as fcti
import numpy as np
import tensorflow as tf
from tf_util.stat_util import approx_equal

DTYPE = tf.float64
DELT_EPS = 1e-4
mus = np.array([-150.0, -1e-6, 0.0, 1.0, 1e-6, 150.0])
delta0s = np.array([0.0, 1e-6, 1.0, 150.0])
deltainfs = np.array([0.0, 1e-6, 1.0, 150.0 - DELT_EPS])

num_mus = mus.shape[0]
num_delta0s = delta0s.shape[0]
num_deltainfs = deltainfs.shape[0]


def eval_single_gaussian_integral(np_func, tf_func, np_num_pts, tf_num_pts, EPS):

    y_true = np.zeros((num_mus, num_delta0s))
    for i in range(num_mus):
        mu = mus[i]
        for j in range(num_delta0s):
            delta0 = delta0s[j]
            y_true[i, j] = np_func(mu, delta0, np_num_pts)

    mu = tf.placeholder(dtype=DTYPE, shape=(num_mus * num_delta0s,))
    delta0 = tf.placeholder(dtype=DTYPE, shape=(num_mus * num_delta0s,))

    _delta0, _mu = np.meshgrid(delta0s, mus)
    _mu = np.reshape(_mu, (num_mus * num_delta0s,))
    _delta0 = np.reshape(_delta0, (num_mus * num_delta0s,))

    y = tf_func(mu, delta0, tf_num_pts)
    y = tf.reshape(y, [num_mus, num_delta0s])

    with tf.Session() as sess:
        _y = sess.run(y, {mu: _mu, delta0: _delta0})

    assert approx_equal(y_true, _y, EPS, perc=False)

    return None


def eval_nested_gaussian_integral(np_func, tf_func, np_num_pts, tf_num_pts, EPS):
    _mu = np.zeros((num_mus * num_delta0s * num_deltainfs,))
    _delta0 = np.zeros((num_mus * num_delta0s * num_deltainfs,))
    _deltainf = np.zeros((num_mus * num_delta0s * num_deltainfs,))

    y_true = np.zeros((num_mus * num_delta0s * num_deltainfs))
    ind = 0
    invalid_inds = []
    for i in range(num_mus):
        mu = mus[i]
        for j in range(num_delta0s):
            delta0 = delta0s[j]
            for k in range(num_deltainfs):
                deltainf = deltainfs[k]
                _mu[ind] = mu
                _delta0[ind] = delta0
                _deltainf[ind] = deltainf
                if deltainf <= delta0:
                    y_true[ind] = np_func(mu, delta0, deltainf, np_num_pts)
                    if np.isnan(y_true[ind]):
                        print("**nan**")
                        print(
                            "mu",
                            _mu[ind],
                            "delta_0",
                            _delta0[ind],
                            "delta_inf",
                            _deltainf[ind],
                        )
                        invalid_inds.append(ind)
                    elif np.isinf(y_true[ind]):
                        print("--inf--")
                        print(
                            "mu",
                            _mu[ind],
                            "delta_0",
                            _delta0[ind],
                            "delta_inf",
                            _deltainf[ind],
                        )
                        invalid_inds.append(ind)
                else:
                    invalid_inds.append(ind)
                ind += 1

    mu = tf.placeholder(dtype=DTYPE, shape=(num_mus * num_delta0s * num_deltainfs,))
    delta0 = tf.placeholder(dtype=DTYPE, shape=(num_mus * num_delta0s * num_deltainfs,))
    deltainf = tf.placeholder(
        dtype=DTYPE, shape=(num_mus * num_delta0s * num_deltainfs,)
    )

    y = tf_func(mu, delta0, deltainf, tf_num_pts)

    with tf.Session() as sess:
        _y = sess.run(y, {mu: _mu, delta0: _delta0, deltainf: _deltainf})

    _y[invalid_inds] = 0.0

    assert approx_equal(y_true, _y, EPS, allow_special=True, perc=False)

    return None


SAME_EPS = 1e-16
COARSE_EPS = 1e-1

ALL_PTS = 200
COARSE_PTS = 100


def test_Prim():
    eval_single_gaussian_integral(fcti.Prim, tfi.Prim, ALL_PTS, ALL_PTS, SAME_EPS)
    eval_single_gaussian_integral(fcti.Prim, tfi.Prim, ALL_PTS, COARSE_PTS, COARSE_EPS)
    return None


def test_Phi():
    eval_single_gaussian_integral(fcti.Phi, tfi.Phi, ALL_PTS, ALL_PTS, SAME_EPS)
    eval_single_gaussian_integral(fcti.Phi, tfi.Phi, ALL_PTS, COARSE_PTS, COARSE_EPS)
    return None


def test_Prime():
    eval_single_gaussian_integral(fcti.Prime, tfi.Prime, ALL_PTS, ALL_PTS, SAME_EPS)
    eval_single_gaussian_integral(
        fcti.Prime, tfi.Prime, ALL_PTS, COARSE_PTS, COARSE_EPS
    )
    return None


def test_Sec():
    eval_single_gaussian_integral(fcti.Sec, tfi.Sec, ALL_PTS, ALL_PTS, SAME_EPS)
    eval_single_gaussian_integral(fcti.Sec, tfi.Sec, ALL_PTS, COARSE_PTS, COARSE_EPS)
    return None


def test_Third():
    eval_single_gaussian_integral(fcti.Third, tfi.Third, ALL_PTS, ALL_PTS, SAME_EPS)
    eval_single_gaussian_integral(
        fcti.Third, tfi.Third, ALL_PTS, COARSE_PTS, COARSE_EPS
    )
    return None


def test_PrimSq():
    eval_single_gaussian_integral(fcti.PrimSq, tfi.PrimSq, ALL_PTS, ALL_PTS, SAME_EPS)
    eval_single_gaussian_integral(
        fcti.PrimSq, tfi.PrimSq, ALL_PTS, COARSE_PTS, COARSE_EPS
    )
    return None


def test_PhiSq():
    eval_single_gaussian_integral(fcti.PhiSq, tfi.PhiSq, ALL_PTS, ALL_PTS, SAME_EPS)
    eval_single_gaussian_integral(
        fcti.PhiSq, tfi.PhiSq, ALL_PTS, COARSE_PTS, COARSE_EPS
    )
    return None


def test_PrimeSq():
    eval_single_gaussian_integral(fcti.PrimeSq, tfi.PrimeSq, ALL_PTS, ALL_PTS, SAME_EPS)
    eval_single_gaussian_integral(
        fcti.PrimeSq, tfi.PrimeSq, ALL_PTS, COARSE_PTS, COARSE_EPS
    )
    return None


def test_PhiPrime():
    eval_single_gaussian_integral(
        fcti.PhiPrime, tfi.PhiPrime, ALL_PTS, ALL_PTS, SAME_EPS
    )
    eval_single_gaussian_integral(
        fcti.PhiPrime, tfi.PhiPrime, ALL_PTS, COARSE_PTS, COARSE_EPS
    )
    return None


def test_PrimPrime():
    eval_single_gaussian_integral(
        fcti.PrimPrime, tfi.PrimPrime, ALL_PTS, ALL_PTS, SAME_EPS
    )
    eval_single_gaussian_integral(
        fcti.PrimPrime, tfi.PrimPrime, ALL_PTS, COARSE_PTS, COARSE_EPS
    )
    return None


def test_PhiSec():
    eval_single_gaussian_integral(fcti.PhiSec, tfi.PhiSec, ALL_PTS, ALL_PTS, SAME_EPS)
    eval_single_gaussian_integral(
        fcti.PhiSec, tfi.PhiSec, ALL_PTS, COARSE_PTS, COARSE_EPS
    )
    return None


def test_PrimPhi():
    eval_single_gaussian_integral(fcti.PrimPhi, tfi.PrimPhi, ALL_PTS, ALL_PTS, SAME_EPS)
    eval_single_gaussian_integral(
        fcti.PrimPhi, tfi.PrimPhi, ALL_PTS, COARSE_PTS, COARSE_EPS
    )
    return None


def test_IntPrimPrim():
    eval_nested_gaussian_integral(
        fcti.IntPrimPrim, tfi.IntPrimPrim, ALL_PTS, ALL_PTS, SAME_EPS
    )
    eval_nested_gaussian_integral(
        fcti.IntPrimPrim, tfi.IntPrimPrim, ALL_PTS, COARSE_PTS, COARSE_EPS
    )
    return None


def test_IntPhiPhi():
    eval_nested_gaussian_integral(
        fcti.IntPhiPhi, tfi.IntPhiPhi, ALL_PTS, ALL_PTS, SAME_EPS
    )
    eval_nested_gaussian_integral(
        fcti.IntPhiPhi, tfi.IntPhiPhi, ALL_PTS, COARSE_PTS, COARSE_EPS
    )
    return None


def test_IntPrimePrime():
    eval_nested_gaussian_integral(
        fcti.IntPrimePrime, tfi.IntPrimePrime, ALL_PTS, ALL_PTS, SAME_EPS
    )
    eval_nested_gaussian_integral(
        fcti.IntPrimePrime, tfi.IntPrimePrime, ALL_PTS, COARSE_PTS, COARSE_EPS
    )
    return None


if __name__ == "__main__":
    test_Prim()
    test_Phi()
    test_Prime()
    test_Sec()
    test_Third()
    test_PrimSq()
    test_PhiSq()
    test_PrimeSq()
    test_PhiPrime()
    test_PrimPrime()
    test_PhiSec()
    test_PrimPhi()
    test_IntPrimPrim()
    test_IntPhiPhi()
    test_IntPrimePrime()
