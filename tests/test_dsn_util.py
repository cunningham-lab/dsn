import tensorflow as tf
import numpy as np
from tf_util.stat_util import approx_equal
from dsn.util.dsn_util import check_convergence

DTYPE = tf.float64
EPS = 1e-16


def test_check_convergence():
    array_len = 1000
    converge_ind = 500

    num_params = 10
    cost_grads = np.zeros((array_len, num_params))
    cost_grads[:converge_ind, :] = np.random.normal(
        2.0, 1.0, (converge_ind, num_params)
    )
    cost_grads[converge_ind:, :] = np.random.normal(
        0.0, 1.0, (converge_ind, num_params)
    )

    lag = 100
    alpha = 0.05
    fail_cur_inds = range(100, 501, 100)
    pass_cur_inds = range(600, 1001, 100)
    for cur_ind in fail_cur_inds:
        assert not check_convergence(cost_grads, cur_ind, lag, alpha)

    for cur_ind in pass_cur_inds:
        assert check_convergence(cost_grads, cur_ind, lag, alpha)

        # All nonzero but one
    cost_grads = np.random.normal(-2.0, 1.0, (array_len, num_params))
    cost_grads[:, 2] = np.random.normal(0.0, 1.0, (array_len,))
    fail_cur_inds = range(100, 1001, 100)
    for cur_ind in fail_cur_inds:
        assert not check_convergence(cost_grads, cur_ind, lag, alpha)

        # All zero but one
    cost_grads = np.random.normal(0.0, 1.0, (array_len, num_params))
    cost_grads[:, 2] = np.random.normal(-2.0, 1.0, (array_len,))
    fail_cur_inds = range(100, 1001, 100)
    for cur_ind in fail_cur_inds:
        assert not check_convergence(cost_grads, cur_ind, lag, alpha)


if __name__ == "__main__":
    test_check_convergence()
