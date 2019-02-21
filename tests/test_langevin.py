import numpy as np
import tensorflow as tf
from dsn.util.tf_langevin import langevin_dyn, langevin_dyn_rank1_spont_static, \
                                 langevin_dyn_rank1_spont_chaos, \
                                 langevin_dyn_rank1_input_chaos
from tf_util.stat_util import approx_equal

EPS = 1e-16
CONV_EPS = 1e-2


def langevin_dyn_np(f, x0, eps, num_its):
    dim = x0.shape[0]
    x = x0
    for i in range(num_its):
        x = (1.0 - eps) * x + eps * f(x)
    return x

n = 200


def test_langevin_dyn():
    x0 = np.random.normal(0.0, 10.0, (n, 3))

    eps = 0.2
    num_iters = 100

    def f(x):
        f1 = x[:, 1] - x[:, 2]
        f2 = x[:, 0] + 3
        f3 = x[:, 1] - 2
        return tf.stack([f1, f2, f3], axis=1)

    def f_np(x):
        f1 = x[1] - x[2]
        f2 = x[0] + 3
        f3 = x[1] - 2
        return np.array([f1, f2, f3])

    x0 = tf.placeholder(dtype=tf.float64, shape=(n, 3))
    _x0 = np.random.normal(0.0, 1.0, (n, 3))

    xs = langevin_dyn(f, x0, eps, num_iters)
    xs_true = np.zeros((n, 3))
    for i in range(n):
        xs_true[i, :] = langevin_dyn_np(f_np, _x0[i, :], eps, num_iters)

    with tf.Session() as sess:
        _xs = sess.run(xs, {x0: _x0})

    x_100_true = np.tile(np.array([[2.0, 5.0, 3.0]]), [n, 1])

    assert approx_equal(_xs, xs_true, EPS)
    assert approx_equal(x_100_true, xs_true, CONV_EPS)
    assert approx_equal(x_100_true, _xs, CONV_EPS)

    # I should come up with some more complex dynamics examples for testing,
    # but for now this is fine.

    return None

def test_langevin_dyn_rank1_spont_static():
    x0 = tf.placeholder(dtype=tf.float64, shape=(n, 2))

    def f(x):
        f1 = x[:, 1]+2
        f2 = 0.0*x[:, 0] + 0.1
        return tf.stack([f1, f2], axis=1)

    eps = 0.8
    num_its = 30

    x_ss, x = langevin_dyn_rank1_spont_static(f, x0, eps, num_its, db=True)

    _x0 = np.random.normal(0.0, 10.0, (n,2))

    with tf.Session() as sess:
        _x = sess.run(x, {x0:_x0})
        
    x_ss_true = np.array([2.1, 0.1])

    for i in range(n):
        assert(approx_equal(_x[i,:,-1], x_ss_true, EPS))
        assert(_x[i,1,1] >= 0.0)
    return None
        

def test_langevin_dyn_rank1_spont_chaos():
    x0 = tf.placeholder(dtype=tf.float64, shape=(n, 3))

    def f(x):
        f1 = x[:, 1]+2
        f2 = 0.0*x[:, 0]+1.1
        f3 = x[:, 1]-1
        return tf.stack([f1, f2, f3], axis=1)

    eps = 0.8
    num_its = 30

    x_ss, x = langevin_dyn_rank1_spont_chaos(f, x0, eps, num_its, db=True)

    _x0 = np.random.normal(0.0, 10.0, (n,3))

    with tf.Session() as sess:
        _x = sess.run(x, {x0:_x0})
        
    x_ss_true = np.array([3.1, 1.1, 0.1])

    for i in range(n):
        assert(approx_equal(_x[i,:,-1], x_ss_true, EPS))
        assert(_x[i,1,1] >= 0.0)
        assert(_x[i,2,1] >= 0.0)
        assert(_x[i,1,1] >= _x[i,2,1])
    return None



def test_langevin_dyn_rank1_input_chaos():
    x0 = tf.placeholder(dtype=tf.float64, shape=(n, 4))

    def f(x):
        f1 = x[:, 2]+2
        f2 = x[:, 2] - 5
        f3 = 0.0*x[:, 0]+1.1
        f4 = x[:, 2]-1
        return tf.stack([f1, f2, f3, f4], axis=1)

    eps = 0.8
    num_its = 30

    x_ss, x = langevin_dyn_rank1_input_chaos(f, x0, eps, num_its, db=True)

    _x0 = np.random.normal(0.0, 10.0, (n,4))

    with tf.Session() as sess:
        _x = sess.run(x, {x0:_x0})
        
    x_ss_true = np.array([3.1, -3.9, 1.1, 0.1])

    for i in range(n):
        assert(approx_equal(_x[i,:,-1], x_ss_true, EPS))
        assert(_x[i,2,1] >= 0.0)
        assert(_x[i,3,1] >= 0.0)
        assert(_x[i,2,1] >= _x[i,3,1])

    return None


if __name__ == "__main__":
    test_langevin_dyn()
    test_langevin_dyn_rank1_spont_static()
    test_langevin_dyn_rank1_spont_chaos()
    test_langevin_dyn_rank1_input_chaos()
