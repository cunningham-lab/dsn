import numpy as np
import tensorflow as tf
from tf_util.stat_util import approx_equal
from dsn.util.tf_DMFT_solvers import rank1_spont_static_solve, rank1_spont_chaotic_solve
import dsn.lib.LowRank.Fig1_Spontaneous.fct_mf as mf
import matplotlib.pyplot as plt

DTYPE = tf.float64

def eval_rank1_spont_static_solve(mu_0, delta0_0, its, langevin_eps, EPS):
	n = 16
	g_start = 0.25
	g_end = 4.0
	_g = np.linspace(g_start, g_end, n)

	# np variables
	_mu_init = mu_0*np.ones((n,))
	_delta_0_init = delta0_0*np.ones((n,))

	_Mm = 1.1*np.ones((n,)) # np.abs(np.random.normal(0.0, 1.0, (n,))) + .001
	_Mn = 2.0*np.ones((n,))
	_Sm = 1.0*np.ones((n,))

	# tf variables
	mu_init = tf.placeholder(dtype=DTYPE, shape=(n,))
	delta_0_init = tf.placeholder(dtype=DTYPE, shape=(n,))

	g = tf.placeholder(dtype=DTYPE, shape=(n,))
	Mm = tf.placeholder(dtype=DTYPE, shape=(n,))
	Mn = tf.placeholder(dtype=DTYPE, shape=(n,))
	Sm = tf.placeholder(dtype=DTYPE, shape=(n,))

	mu_true = np.zeros((n,))
	delta_0_true = np.zeros((n,))

	for k in range(n):
		y0_k = [_mu_init[k], _delta_0_init[k]]
		VecPar_k = [_Mm[k], _Mn[k], _Sm[k], 0.0]
		y = mf.SolveStatic(y0_k, _g[k], VecPar_k, tolerance=1e-8, backwards=1)
		mu_true[k] = y[0]
		delta_0_true[k] = y[1]

	mu, delta_0 = rank1_spont_static_solve(mu_init, delta_0_init, g, Mm, Mn, Sm, its, langevin_eps)

	feed_dict = {mu_init:_mu_init, delta_0_init:_delta_0_init, g:_g, Mm:_Mm, Mn:_Mn, Sm:_Sm}
	with tf.Session() as sess:
		_mu, _delta_0 = sess.run([mu, delta_0], feed_dict)

	assert(approx_equal(_mu, mu_true, EPS))
	assert(approx_equal(_delta_0, delta_0_true, EPS))


	return None

def eval_rank1_spont_chaotic_solve(mu_0, delta0_0, deltainf_0, its, langevin_eps, EPS):
	n = 16
	g_start = 0.25
	g_end = 4.0
	_g = np.linspace(g_start, g_end, n)

	# np variables
	_mu_init = mu_0*np.ones((n,))
	_delta_0_init = delta0_0*np.ones((n,))
	_delta_inf_init = deltainf_0*np.ones((n,))

	_Mm = 1.1*np.ones((n,)) # np.abs(np.random.normal(0.0, 1.0, (n,))) + .001
	_Mn = 2.0*np.ones((n,))
	_Sm = 1.0*np.ones((n,))

	# tf variables
	mu_init = tf.placeholder(dtype=DTYPE, shape=(n,))
	delta_0_init = tf.placeholder(dtype=DTYPE, shape=(n,))
	delta_inf_init = tf.placeholder(dtype=DTYPE, shape=(n,))

	g = tf.placeholder(dtype=DTYPE, shape=(n,))
	Mm = tf.placeholder(dtype=DTYPE, shape=(n,))
	Mn = tf.placeholder(dtype=DTYPE, shape=(n,))
	Sm = tf.placeholder(dtype=DTYPE, shape=(n,))

	mu_true = np.zeros((n,))
	delta_0_true = np.zeros((n,))
	delta_inf_true = np.zeros((n,))

	for k in range(n):
		y0_k = [_mu_init[k], _delta_0_init[k], _delta_inf_init[k]]
		VecPar_k = [_Mm[k], _Mn[k], _Sm[k], 0.0]
		y = mf.SolveChaotic(y0_k, _g[k], VecPar_k, tolerance=1e-8, backwards=1)
		mu_true[k] = y[0]
		delta_0_true[k] = y[1]
		delta_inf_true[k] = y[2]

	mu, delta_0, delta_inf = rank1_spont_chaotic_solve(mu_init, delta_0_init, delta_inf_init, \
		                                         g, Mm, Mn, Sm, its, langevin_eps, gauss_quad_pts=50)

	feed_dict = {mu_init:_mu_init, delta_0_init:_delta_0_init, delta_inf_init:_delta_inf_init, \
	             g:_g, Mm:_Mm, Mn:_Mn, Sm:_Sm}
	with tf.Session() as sess:
		_mu, _delta_0, _delta_inf = sess.run([mu, delta_0, delta_inf], feed_dict)

	assert(approx_equal(_mu, mu_true, EPS))
	assert(approx_equal(_delta_0, delta_0_true, EPS))
	assert(approx_equal(_delta_inf, delta_inf_true, EPS))


	return None

def test_rank1_spont_static_solve():
	print('static')
	FINE_EPS = 1e-6
	COARSE_EPS = 1e-2
	mu_inits = [50.0, -50.0]
	delta0_init = 50.0
	for mu_init in mu_inits:
		print(mu_init)
		print(150)
		eval_rank1_spont_static_solve(mu_init, delta0_init, 150, 0.8, FINE_EPS)
		print(30)
		eval_rank1_spont_static_solve(mu_init, delta0_init, 30, 0.8, COARSE_EPS)


def test_rank1_spont_chaotic_solve():
	print('chaotic')
	"""
	Yeah, so the plots from F&O 2018 have early stopping criteria for when 
	deltainf gets greater than delta0.  In our code, we enforce the 
	constraints on each langevin dynamics iteration, allowing e.g. convergence
	of mu past the point where those two variables become equal.  I think
	this is what shuld really be plotted in those figures.  At the end of the 
	day, this can be reseolved in finite-size simulations.
	"""
	FINE_EPS = 1.0
	COARSE_EPS = 1.0
	mu_inits = [50.0, -50.0]
	delta0_init = 55.0
	deltainf_init = 45.0
	for mu_init in mu_inits:
		print(mu_init)
		print(200)
		eval_rank1_spont_chaotic_solve(mu_init, delta0_init, deltainf_init, \
			                           200, 0.2, FINE_EPS)
		print(30)
		eval_rank1_spont_chaotic_solve(mu_init, delta0_init, deltainf_init, \
		                               30, 0.8, COARSE_EPS)

if __name__ == "__main__":
    test_rank1_spont_static_solve()
    test_rank1_spont_chaotic_solve()
    
