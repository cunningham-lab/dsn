import numpy as np
import tensorflow as tf
from tf_util.stat_util import approx_equal
from dsn.util.tf_DMFT_solvers import (
    rank1_spont_static_solve, 
    rank1_spont_static_solve_np, 
    rank2_CDD_static_solve,
    rank2_CDD_static_solve_np,
)
import dsn.lib.LowRank.Fig1_Spontaneous.fct_mf as mf
import matplotlib.pyplot as plt

DTYPE = tf.float64

EPS = 1e-16
langevin_eps = 0.2
its = 100


def test_rank1_spont_static_solve():
    n = 1000
    _g = np.random.uniform(0.01, 5.0, n)

    # np variables
    _mu_init = np.random.uniform(0.01, 150.0, n)
    _delta_0_init = np.random.uniform(0.01, 150.0, n)

    _Mm = np.random.normal(-5.0, 5.0, (n,))
    _Mn = np.random.normal(-5.0, 5.0, (n,))
    _Sm = np.random.uniform(0.01, 5.0, n)

    # tf variables
    mu_init = tf.placeholder(dtype=DTYPE, shape=(n,))
    delta_0_init = tf.placeholder(dtype=DTYPE, shape=(n,))

    g = tf.placeholder(dtype=DTYPE, shape=(n,))
    Mm = tf.placeholder(dtype=DTYPE, shape=(n,))
    Mn = tf.placeholder(dtype=DTYPE, shape=(n,))
    Sm = tf.placeholder(dtype=DTYPE, shape=(n,))

    mu, delta_0 = rank1_spont_static_solve(
        mu_init, delta_0_init, g, Mm, Mn, Sm, its, langevin_eps, gauss_quad_pts=200
    )
    mu_np, delta_0_np = rank1_spont_static_solve_np(
        _mu_init, _delta_0_init, _g, _Mm, _Mn, _Sm, its, langevin_eps
    )

    feed_dict = {
        mu_init: _mu_init,
        delta_0_init: _delta_0_init,
        g: _g,
        Mm: _Mm,
        Mn: _Mn,
        Sm: _Sm,
    }
    with tf.Session() as sess:
        _mu, _delta_0 = sess.run([mu, delta_0], feed_dict)

    assert approx_equal(_mu, mu_np, 1e-6)
    assert approx_equal(_delta_0, delta_0_np, 1e-6)

    return None

def test_rank2_CDD_static_solve():
    n = 1000
    _g = np.random.uniform(0.01, 5.0, n)

    # np variables
    _kappa1_init = np.random.uniform(-150.0, 0.0, n)
    _kappa2_init = np.random.uniform(-150.0, 0.0, n)
    _delta_0_init = np.random.uniform(0.01, 150.0, n)

    _g = np.random.uniform(0.01, 5.0, n)
    _cA = np.random.uniform(-5.0, 5.0, (n,))
    _cB = np.random.uniform(-5.0, 5.0, (n,))
    _rhom = np.random.uniform(-5.0, 5.0, (n,))
    _rhon = np.random.uniform(-5.0, 5.0, (n,))
    _betam = np.random.uniform(-5.0, 5.0, (n,))
    _betan = np.random.uniform(-5.0, 5.0, (n,))
    _gammaA = np.random.uniform(-5.0, 5.0, (n,))
    _gammaB = np.random.uniform(-5.0, 5.0, (n,))

    # tf variables
    kappa1_init = tf.placeholder(dtype=DTYPE, shape=(n,))
    kappa2_init = tf.placeholder(dtype=DTYPE, shape=(n,))
    delta_0_init = tf.placeholder(dtype=DTYPE, shape=(n,))

    g = tf.placeholder(dtype=DTYPE, shape=(n,))
    cA = tf.placeholder(dtype=DTYPE, shape=(n,))
    cB = tf.placeholder(dtype=DTYPE, shape=(n,))
    rhom = tf.placeholder(dtype=DTYPE, shape=(n,))
    rhon = tf.placeholder(dtype=DTYPE, shape=(n,))
    betam = tf.placeholder(dtype=DTYPE, shape=(n,))
    betan = tf.placeholder(dtype=DTYPE, shape=(n,))
    gammaA = tf.placeholder(dtype=DTYPE, shape=(n,))
    gammaB = tf.placeholder(dtype=DTYPE, shape=(n,))

    kappa1, kappa2, delta_0, z = rank2_CDD_static_solve(
        kappa1_init, kappa2_init, delta_0_init, 
        cA, cB, g, rhom, rhon, betam, betan, gammaA, gammaB,
        its, langevin_eps, gauss_quad_pts=200
    )
    kappa1_np, kappa2_np, delta_0_np, z_np = rank2_CDD_static_solve_np(
        _kappa1_init, _kappa2_init, _delta_0_init, 
        _cA, _cB, _g, _rhom, _rhon, _betam, _betan, _gammaA, _gammaB,
        its, langevin_eps
    )

    feed_dict = {
        kappa1_init: _kappa1_init,
        kappa2_init: _kappa2_init,
        delta_0_init: _delta_0_init,
        cA: _cA,
        cB: _cB,
        g: _g,
        rhom: _rhom,
        rhon: _rhon,
        betam: _betam,
        betan: _betan,
        gammaA: _gammaA,
        gammaB: _gammaB,
    }
    with tf.Session() as sess:
        _kappa1, _kappa2, _delta_0, _z = sess.run([kappa1, kappa2, delta_0, z], feed_dict)

    assert approx_equal(_kappa1, kappa1_np, 1e-6)
    assert approx_equal(_kappa2, kappa2_np, 1e-6)
    assert approx_equal(_delta_0, delta_0_np, 1e-6)
    assert approx_equal(_z, z_np, 1e-6)

    return None


if __name__ == "__main__":
    test_rank1_spont_static_solve()
    test_rank2_CDD_static_solve()
