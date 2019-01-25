import numpy as np
import tensorflow as tf
from dsn.util.tf_langevin import langevin_dyn
from tf_util.stat_util import approx_equal

EPS = 1e-16
CONV_EPS = 1e-2

def langevin_dyn_np(f, x0, eps, num_its):
	dim = x0.shape[0]
	xs = np.zeros((dim, num_its+1))
	xs[:,0] = x0
	for i in range(num_its):
		xs[:,i+1] = (1.0-eps)*xs[:,i] + eps*f(xs[:,i])
	return xs


def test_langevin_dyn():
	n = 200
	x0 = np.random.normal(0.0, 10.0, (n, 3))

	eps = 0.2
	num_iters = 100

	def f(x):
		f1 = x[:,1] - x[:,2]
		f2 =  x[:,0] + 3
		f3 = x[:,1] - 2
		return tf.stack([f1, f2, f3], axis=1)

	def f_np(x):
		f1 = x[1] - x[2]
		f2 =  x[0] + 3
		f3 = x[1] - 2
		return np.array([f1, f2, f3])

	x0 = tf.placeholder(dtype=tf.float64, shape=(n,3))
	_x0 = np.random.normal(0.0, 1.0, (n, 3))

	xs = langevin_dyn(f, x0, eps, num_iters)
	xs_true = np.zeros((n, 3, num_iters+1))
	for i in range(n):
		xs_true[i,:,:] = langevin_dyn_np(f_np, _x0[i,:], eps, num_iters)

	with tf.Session() as sess:
		_xs = sess.run(xs, {x0:_x0})

	assert(approx_equal(_xs, xs_true, EPS))
	x_100_true = np.tile(np.array([[2.0, 5.0, 3.0]]),[n, 1])
	assert(approx_equal(x_100_true, xs_true[:,:,-1], CONV_EPS))
	assert(approx_equal(x_100_true, _xs[:,:,-1], CONV_EPS))


	# I should come up with some more complex dynamics examples for testing,
	# but for now this is fine.

	return None

if __name__ == "__main__":
	test_langevin_dyn()

