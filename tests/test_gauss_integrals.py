import dsn.util.tf_integrals as tfi
import dsn.util.fct_integrals as fcti
import numpy as np
import tensorflow as tf
from tf_util.stat_util import approx_equal

DTYPE = tf.float64
EPS = 1e-16

mus = np.array([-100.0, -1.0, 1e-6, 0.0, 1e-6, 1.0, 100.0])
delta0s = np.array([0.0, 1e-6, 1.0, 100.0])
deltainfs = np.array([0.0, 1e-6, 1.0, 100.0])

num_mus = mus.shape[0]
num_delta0s = delta0s.shape[0]
num_deltainfs = deltainfs.shape[0]

def eval_single_gaussian_integral(fct_fun, tf_func):

	y_true = np.zeros((num_mus, num_delta0s))
	for i in range(num_mus):
		mu = mus[i]
		for j in range(num_delta0s):
			delta0 = delta0s[j]
			y_true[i,j] = fct_fun(mu, delta0)

	mu = tf.placeholder(dtype=DTYPE, shape=(num_mus*num_delta0s,))
	delta0 = tf.placeholder(dtype=DTYPE, shape=(num_mus*num_delta0s,))

	_delta0, _mu = np.meshgrid(delta0s, mus)
	_mu = np.reshape(_mu, (num_mus*num_delta0s,))
	_delta0 = np.reshape(_delta0, (num_mus*num_delta0s,))

	y = tf_func(mu, delta0)
	y = tf.reshape(y, [num_mus, num_delta0s])

	with tf.Session() as sess:
		_y = sess.run(y, {mu:_mu, delta0:_delta0})

	assert(approx_equal(y_true, _y, EPS))

	return None

def eval_nested_gaussian_integral(fct_fun, tf_func):
	_mu = np.zeros((num_mus*num_delta0s*num_deltainfs,))
	_delta0 = np.zeros((num_mus*num_delta0s*num_deltainfs,))
	_deltainf = np.zeros((num_mus*num_delta0s*num_deltainfs,))

	y_true = np.zeros((num_mus*num_delta0s*num_deltainfs))
	ind = 0
	for i in range(num_mus):
		mu = mus[i]
		for j in range(num_delta0s):
			delta0 = delta0s[j]
			for k in range(num_deltainfs):
				deltainf = deltainfs[k]
				_mu[ind] = mu
				_delta0[ind] = delta0
				_deltainf[ind] = deltainf
				y_true[ind] = fct_fun(mu, delta0, deltainf)
				ind += 1

	mu = tf.placeholder(dtype=DTYPE, shape=(num_mus*num_delta0s*num_deltainfs,))
	delta0 = tf.placeholder(dtype=DTYPE, shape=(num_mus*num_delta0s*num_deltainfs,))
	deltainf = tf.placeholder(dtype=DTYPE, shape=(num_mus*num_delta0s*num_deltainfs,))

	y = tf_func(mu, delta0, deltainf)
	#y = tf.reshape(y, [num_mus, num_delta0s, num_deltainfs])

	with tf.Session() as sess:
		_y = sess.run(y, {mu:_mu, delta0:_delta0, deltainf:_deltainf})

	assert(approx_equal(y_true, _y, EPS, allow_special=True))

	return None

def test_Prim():
	eval_single_gaussian_integral(fcti.Prim, tfi.Prim)
	return None

def test_Phi():
	eval_single_gaussian_integral(fcti.Phi, tfi.Phi)
	return None

def test_Prime():
	eval_single_gaussian_integral(fcti.Prime, tfi.Prime)
	return None

def test_Sec():
	eval_single_gaussian_integral(fcti.Sec, tfi.Sec)
	return None

def test_Third():
	eval_single_gaussian_integral(fcti.Third, tfi.Third)
	return None

def test_PrimSq():
	eval_single_gaussian_integral(fcti.PrimSq, tfi.PrimSq)
	return None

def test_PhiSq():
	eval_single_gaussian_integral(fcti.PhiSq, tfi.PhiSq)
	return None

def test_PrimeSq():
	eval_single_gaussian_integral(fcti.PrimeSq, tfi.PrimeSq)
	return None

def test_PhiPrime():
	eval_single_gaussian_integral(fcti.PhiPrime, tfi.PhiPrime)
	return None

def test_PrimPrime():
	eval_single_gaussian_integral(fcti.PrimPrime, tfi.PrimPrime)
	return None

def test_PhiSec():
	eval_single_gaussian_integral(fcti.PhiSec, tfi.PhiSec)
	return None

def test_PrimPhi():
	eval_single_gaussian_integral(fcti.PrimPhi, tfi.PrimPhi)
	return None

def test_IntPrimPrim():
	eval_nested_gaussian_integral(fcti.IntPrimPrim, tfi.IntPrimPrim)
	return None

def test_IntPhiPhi():
	eval_nested_gaussian_integral(fcti.IntPhiPhi, tfi.IntPhiPhi)
	return None

def test_IntPrimePrime():
	eval_nested_gaussian_integral(fcti.IntPrimePrime, tfi.IntPrimePrime)
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
