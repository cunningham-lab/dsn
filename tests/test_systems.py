import tensorflow as tf
import numpy as np
import scipy
from tf_util.stat_util import approx_equal
from dsn.util.systems import system, Linear2D, V1Circuit, LowRankRNN
import matplotlib.pyplot as plt
import dsn.lib.LowRank.Fig1_Spontaneous.fct_mf as mf

DTYPE = tf.float64
EPS = 1e-16

class linear_2D:
	def __init__(self,):
		pass

	def compute_mu(self, behavior):
		means = behavior['means']
		variances = behavior['variances']
		first_mom = means
		second_mom = means**2 + variances
		return np.concatenate((first_mom, second_mom), axis=0)

	def compute_suff_stats(self, tau, A):
		C = A  / float(tau)
		c1 = C[0,0]
		c2 = C[0,1]
		c3 = C[1,0]
		c4 = C[1,1]

		root_term = np.square(c1 + c4) - 4*(c1*c4 - c2*c3)
		if (root_term >= 0):
			lambda_1_real = (c1 + c4 + np.sqrt(root_term)) / 2.0
			lambda_1_imag = 0.0
		else:
			lambda_1_real = (c1 + c4) / 2.0
			lambda_1_imag = np.sqrt(-root_term) / 2.0

		T_x = np.array([lambda_1_real, lambda_1_imag, \
			            lambda_1_real**2, lambda_1_imag**2])
		return T_x

class v1_circuit():
	def __init__(self, T=40, dt=0.25, init_conds=np.array([[1.0], [1.1], [1.2], [1.3]])):
		self.init_conds = init_conds
		self.dt = dt
		self.T = T

	def compute_h(self, c, s, r, b, h_FF, h_LAT, h_RUN, \
		          g_FF_str, g_LAT_str, g_RUN_str, \
		          s_0, a=None, c_50=None):
		# Set the gates
		if (g_FF_str == 'c'):
			g_FF = c
		elif (g_FF_str == 'saturate'):
			assert(a is not None)
			assert(c_50 is not None)
			g_FF = (c**a) / (c_50**a + c**a)
		else:
			raise NotImplementedError()

		if (g_LAT_str == 'linear'):
			s_diff = s_0 - s
		elif (g_LAT_str == 'square'):
			s_diff = s_0**2 - s**2
		else:
			raise NotImplementedError()
		if (s_diff < 0):
			g_LAT = 0.0
		else:
			g_LAT = c*s_diff

		if (g_RUN_str== 'r'):
			g_RUN = r

		h = b + g_FF*h_FF + g_LAT*h_LAT + g_RUN*h_RUN
		return h

	def simulate(self, W, h, tau, n):
		r0 = self.init_conds[:,0]

		def f(r, t):
			pow_arg = np.dot(W, np.expand_dims(r, 1)) + h

			for i in range(4):
				if (pow_arg[i,0] < 0):
					pow_arg[i,0] = 0.0

			drdt = (1.0/tau) * (-r + np.power(pow_arg[:,0], n))

			for i in range(4):
				if (drdt[i] > 1e6):
					drdt[i] = 1e6
				if (drdt[i] < -1e6):
					drdt[i] = -1e6

			return drdt

		t = np.arange(0, self.T*self.dt, self.dt)
		y = scipy.integrate.odeint(f, r0, t)

		return y


class low_rank_rnn:
	def __init__(self,):
		pass

	def compute_mu(self, behavior):
		means = behavior['means']
		variances = behavior['variances']
		first_mom = means
		second_mom = means**2 + variances
		return np.concatenate((first_mom, second_mom), axis=0)

	def compute_suff_stats(self, g, Mm, Mn, Sm):
		y0 = [50.0, 55.0, 45.0]
		VecPar = [Mm, Mn, Sm, 0.0]
		y = mf.SolveChaotic(y0, g, VecPar, tolerance=1e-4, backwards=1)
		mu = y[0]
		delta_0 = y[1]
		delta_inf = y[2]
		return np.array([mu, delta_inf, delta_0-delta_inf, \
			             mu**2, delta_inf**2, (delta_0-delta_inf)**2])



n = 100

def test_Linear2D():
	with tf.Session() as sess:
		true_sys = linear_2D()
		Z = tf.placeholder(dtype=DTYPE, shape=(1, n, None))

		omega = 2;
		mu1 = np.array([0.0, 2*np.pi*omega]);
		Sigma1 = np.array([1.0, 1.0]);
		behavior1 = {'type':'oscillation', 'means':mu1, 'variances':Sigma1};

		# no fixed parameters
		sys = Linear2D({}, behavior1)
		assert(sys.name == 'Linear2D')
		assert(not sys.fixed_params) # empty dict evaluates to False
		assert(sys.all_params == ['A', 'tau'])
		assert(sys.free_params == ['A', 'tau'])
		assert(sys.z_labels == ['$a_1$', '$a_2$', '$a_3$', '$a_4$', '$\\tau$'])
		assert(sys.T_x_labels == [r'real($\lambda_1$)', r'imag($\lambda_1$)', \
	                          r'real$(\lambda_1)^2$', r'imag$(\lambda_1)^2$'])
		assert(sys.D == 5)
		assert(sys.num_suff_stats == 4)

		# Fix tau to 1.0
		tau = 1.0
		sys = Linear2D({'tau':tau}, behavior1)
		assert(sys.fixed_params['tau'] == tau) # empty dict evaluates to False
		assert(sys.all_params == ['A', 'tau'])
		assert(sys.free_params == ['A'])
		assert(sys.z_labels == ['$a_1$', '$a_2$', '$a_3$', '$a_4$'])
		assert(sys.T_x_labels == [r'real($\lambda_1$)', r'imag($\lambda_1$)', \
	                          r'real$(\lambda_1)^2$', r'imag$(\lambda_1)^2$'])
		assert(sys.D == 4)
		assert(sys.num_suff_stats == 4)

		# Fix A to eye(2)
		A = np.eye(2)
		sys = Linear2D({'A':A}, behavior1)
		assert(approx_equal(sys.fixed_params['A'], A, EPS)) # empty dict evaluates to False
		assert(sys.all_params == ['A', 'tau'])
		assert(sys.free_params == ['tau'])
		assert(sys.z_labels == ['$\\tau$'])
		assert(sys.T_x_labels == [r'real($\lambda_1$)', r'imag($\lambda_1$)', \
	                           r'real$(\lambda_1)^2$', r'imag$(\lambda_1)^2$'])
		assert(sys.D == 1)
		assert(sys.num_suff_stats == 4)

		# Fix tau to 1.0 and A to eye(2)
		A = np.eye(2)
		sys = Linear2D({'A':A, 'tau':tau}, behavior1)
		assert(sys.fixed_params['tau'] == tau)
		assert(approx_equal(sys.fixed_params['A'], A, EPS)) # empty dict evaluates to False
		assert(sys.all_params == ['A', 'tau'])
		assert(sys.free_params == [])
		assert(sys.z_labels == [])
		assert(sys.T_x_labels == [r'real($\lambda_1$)', r'imag($\lambda_1$)', \
	                           r'real$(\lambda_1)^2$', r'imag$(\lambda_1)^2$'])
		assert(sys.D == 0)
		assert(sys.num_suff_stats == 4)


		# test mu computation
		sys = Linear2D({}, behavior1)
		mu1 = true_sys.compute_mu(behavior1)
		assert(approx_equal(sys.mu, mu1, EPS))

		mu2 = np.array([4.0, 4.0]);
		Sigma2 = np.array([0.001, 1000]);
		behavior2 = {'type':'oscillation', 'means':mu2, 'variances':Sigma2};

		mu3 = np.array([-4.0, -4.0]);
		Sigma3 = np.array([1e-7, 1e7]);
		behavior3 = {'type':'oscillation', 'means':mu3, 'variances':Sigma3};

		sys = Linear2D({}, behavior2)
		mu2 = true_sys.compute_mu(behavior2)
		assert(approx_equal(sys.mu, mu2, EPS))

		sys = Linear2D({}, behavior3)
		mu3 = true_sys.compute_mu(behavior3)
		assert(approx_equal(sys.mu, mu3, EPS))


		# test sufficient statistic computation
		sys = Linear2D({}, behavior1)
		T_x = sys.compute_suff_stats(Z)
		tau = np.abs(np.random.normal(0.0, 10.0, (1,n,1))) + 0.001
		A = np.random.normal(0.0, 10.0, (1,n,2,2))
		_Z = np.concatenate((np.reshape(A, [1, n, 4]), tau), axis=2)
		_T_x_true = np.zeros((n,sys.num_suff_stats))
		for i in range(n):
			_T_x_true[i,:] = true_sys.compute_suff_stats(tau[0,i], A[0,i])
		_T_x = sess.run(T_x, {Z:_Z})
		assert(approx_equal(_T_x[0,:,:], _T_x_true, EPS))

	print('Linear2D passed.')
	return None


def test_V1Circuit():
	np.random.seed(0)
	M = 100
	sess = tf.Session()
	dt = 0.25
	T = 40
	true_sys = v1_circuit(T=T, dt=dt)
	Z = tf.placeholder(dtype=DTYPE, shape=(1, M, None))

	# difference behavior 1
	c_vals=np.array([0.0])
	s_vals=np.array([1.0])
	r_vals=np.array([0.0, 1.0])
	d_mean = np.zeros((4,))
	d_var = np.ones((4,))
	behavior1 = {'type':'difference', \
				'c_vals':c_vals, \
				's_vals':s_vals, \
				'r_vals':r_vals, \
				'd_mean':d_mean, \
				'd_var':d_var}

	# ****************************************
	# One random fixed param
	# ****************************************

	b_S = 1.0
	fixed_params = {'b_S':b_S}
	system = V1Circuit(fixed_params, behavior1, T=T, dt=dt)
	# Test free/fixed parameter handling
	assert(system.name == 'V1Circuit')
	#assert(system.T == 40)
	#assert(system.dt == 0.25)
	assert(approx_equal(system.init_conds, np.array([[1.0], [1.1], [1.2], [1.3]]), EPS))
	assert(system.model_opts['g_FF'] == 'c')
	assert(system.model_opts['g_LAT'] == 'linear')
	assert(system.model_opts['g_RUN'] == 'r')
	assert(approx_equal(system.behavior['c_vals'], c_vals, EPS))
	assert(approx_equal(system.behavior['s_vals'], s_vals, EPS))
	assert(approx_equal(system.behavior['r_vals'], r_vals, EPS))
	assert(approx_equal(system.behavior['d_mean'], d_mean, EPS))
	assert(approx_equal(system.behavior['d_var'], d_var, EPS))
	assert(system.C == 2)
	assert(system.fixed_params['b_S'] == 1.0)
	assert(len(system.fixed_params.keys()) == 1)
	assert(system.all_params == ['W_EE', 'W_PE', 'W_SE', 'W_VE', \
								'b_E', 'b_P', 'b_S', 'b_V', \
								'h_FFE', 'h_FFP', \
								'h_LATE', 'h_LATP', 'h_LATS', 'h_LATV', \
								'h_RUNE', 'h_RUNP', 'h_RUNS', 'h_RUNV', \
								'tau', 'n', 's_0'])
	assert(system.free_params == ['W_EE', 'W_PE', 'W_SE', 'W_VE', \
										'b_E', 'b_P', 'b_V', \
										'h_FFE', 'h_FFP', \
										'h_LATE', 'h_LATP', 'h_LATS', 'h_LATV', \
										'h_RUNE', 'h_RUNP', 'h_RUNS', 'h_RUNV', \
										'tau', 'n', 's_0'])
	assert(system.z_labels == [r'$W_{EE}$', r'$W_{PE}$', r'$W_{SE}$', r'$W_{VE}$', \
                      r'$b_{E}$', r'$b_{P}$', r'$b_{V}$', \
                      r'$h_{FF,E}$', r'$h_{FF,P}$', \
                      r'$h_{LAT,E}$', r'$h_{LAT,P}$', r'$h_{LAT,S}$', r'$h_{LAT,V}$', \
                      r'$h_{RUN,E}$', r'$h_{RUN,P}$', r'$h_{RUN,S}$', r'$h_{RUN,V}$', \
                      r'$\tau$', r'$n$', r'$s_0$'])
	assert(system.T_x_labels == [r'$d_{E,ss}$', r'$d_{P,ss}$', r'$d_{S,ss}$', r'$d_{V,ss}$', \
								r'$d_{E,ss}^2$', r'$d_{P,ss}^2$', r'$d_{S,ss}^2$', r'$d_{V,ss}^2$'])
	assert(system.D == 20)
	assert(system.num_suff_stats == 8)

	# Test density network output filtering
	Z = tf.placeholder(dtype=tf.float64, shape=(1, M, system.D))
	W, b, h_FF, h_LAT, h_RUN, tau, n, s_0, a, c_50 = system.filter_Z(Z)
	assert(a is None)
	assert(c_50 is None)

	_Z = np.random.normal(0.0, 1.0, (1, M, system.D))
	_Z[:,:,:4] = np.abs(_Z[:,:,:4])
	_Z[:,:,17:19] = np.abs(_Z[:,:,17:19])
	_W, _b, _h_FF, _h_LAT, _h_RUN, _tau, _n, _s_0 = sess.run([W, b, h_FF, h_LAT, h_RUN, tau, n, s_0], {Z:_Z})
	assert(_W.shape == (2,M,4,4))
	assert(_b.shape == (1,M,4,1))
	assert(_h_FF.shape == (1,M,4,1))
	assert(_h_LAT.shape == (1,M,4,1))
	assert(_h_RUN.shape == (1,M,4,1))
	assert(_tau.shape == (2,M,1,1))
	assert(_n.shape == (2,M,1,1))
	assert(_s_0.shape == (1,M,1,1))
	W_PSV = np.tile(np.array([[[-1.0,  -0.54, 0.0], \
							  [-1.01, -0.33, 0.0], \
							  [0.0,   0.0, -0.15], \
							  [-0.22, -0.77, 0.0]]]), [M, 1, 1])
	W_E = np.expand_dims(_Z[0,:,:4], 2)
	W_true = np.concatenate((W_E, W_PSV), axis=2)
	assert(approx_equal(_W[0], W_true, EPS))
	assert(approx_equal(_W[1], W_true, EPS))
	b_true = np.expand_dims(np.concatenate((_Z[:,:,4:6], fixed_params['b_S']*np.ones((1,M,1)), _Z[:,:,6:7]), axis=2), 3)
	assert(approx_equal(_b, b_true, EPS))
	h_FF_true = np.concatenate((np.expand_dims(_Z[:,:,7:9], 3), np.zeros((1, M, 2, 1))), axis=2)
	assert(approx_equal(_h_FF, h_FF_true, EPS))
	h_LAT_true = np.expand_dims(_Z[:,:,9:13], 3)
	assert(approx_equal(_h_LAT, h_LAT_true, EPS))
	h_RUN_true = np.expand_dims(_Z[:,:,13:17], 3)
	assert(approx_equal(_h_RUN, h_RUN_true, EPS))
	tau_true = np.tile(np.expand_dims(_Z[:,:,17:18], 3), [2,1,1,1])
	assert(approx_equal(_tau, tau_true, EPS))
	n_true = np.tile(np.expand_dims(_Z[:,:,18:19], 3), [2,1,1,1])
	assert(approx_equal(_n, n_true, EPS))
	s_0_true = np.expand_dims(_Z[:,:,19:20], 3)
	assert(approx_equal(_s_0, s_0_true, EPS))

	# Test compute h (input)
	h = system.compute_h(b, h_FF, h_LAT, h_RUN, s_0, a, s_0)
	_h = sess.run(h, {Z:_Z})

	h_true = np.zeros((2,M,4,1))
	for i in range(M):
		h_true[0,i,:,:] = true_sys.compute_h(0.0, 0.0, 0.0, b_true[0,i,:,:], \
			                                 h_FF_true[0,i,:,:], h_LAT_true[0,i,:,:], h_RUN_true[0,i,:,:], \
			                                 system.model_opts['g_FF'], system.model_opts['g_LAT'], system.model_opts['g_RUN'], \
		                                     s_0_true[0,i,0,0], None, None)
		h_true[1,i,:,:] = true_sys.compute_h(0.0, 0.0, 1.0, b_true[0,i,:,:], \
			                                 h_FF_true[0,i,:,:], h_LAT_true[0,i,:,:], h_RUN_true[0,i,:,:], \
			                                 system.model_opts['g_FF'], system.model_opts['g_LAT'], system.model_opts['g_RUN'], \
		                                     s_0_true[0,i,0,0], None, None)
	assert(approx_equal(_h, h_true, EPS))

	# Test simulation
	r_t = system.simulate(Z)
	r_t_true = np.zeros((2,M,4,system.T))
	for c in range(2):
		for i in range(M):
			if (c==0 and i==5):
				print('c', c, 'i', i)
				print('W')
				print(W_true[i,:,:])
				print('h')
				print(h_true[c,i,:,:])
				print('tau')
				print(tau_true[c,i,0,0])
				print('n')
				print(n_true[c,i,0,0])
				print('Wr + h')
				_r = system.init_conds
				power_arg = np.dot(W_true[i,:,:], _r) + h_true[c,i,:,:]
				print(power_arg)
				print('(Wr+h)+^n')
				for ii in range(4):
					if (power_arg[ii,0] < 0):
						power_arg[ii,0] = 0.0
				print(power_arg**n_true[c,i,0,0])
				print(np.power(power_arg, n_true[c,i,0,0]))
				print('-r + (Wr+h)+^n')
				print(-_r + np.power(power_arg, n_true[c,i,0,0]))
				print('(1/tau) * (-r + (Wr+h)+^n)')
				print((1.0/tau_true[c,i,0,0])*(-_r + np.power(power_arg, n_true[c,i,0,0])))

			r_t_true[c,i,:,:] = true_sys.simulate(W_true[i,:,:], h_true[c,i,:,:], tau_true[c,i,0,0], n_true[c,i,0,0]).T

	_r_t = sess.run(r_t, {Z:_Z}) # [T, C, M, system.D, 1]
	_r_t = np.transpose(_r_t[:,:,:,:,0], [1, 2, 3, 0])
	"""
	for i in range(M):
		print(i)
		plt.figure()
		plt.plot(_r_t[0,i,0,:], 'k')
		plt.plot(r_t_true[0,i,0,:], 'k--')
		plt.plot(_r_t[0,i,1,:], 'b')
		plt.plot(r_t_true[0,i,1,:], 'b--')
		plt.plot(_r_t[0,i,2,:], 'g')
		plt.plot(r_t_true[0,i,2,:], 'g--')
		plt.plot(_r_t[0,i,3,:], 'r')
		plt.plot(r_t_true[0,i,3,:], 'r--')
		plt.legend(['tf', 'np'])
		plt.title('c %d i %d' % (0, i))
		plt.show()
		plt.figure()
		plt.plot(_r_t[1,i,0,:], 'k')
		plt.plot(r_t_true[1,i,0,:], 'k--')
		plt.plot(_r_t[1,i,1,:], 'b')
		plt.plot(r_t_true[1,i,1,:], 'b--')
		plt.plot(_r_t[1,i,2,:], 'g')
		plt.plot(r_t_true[1,i,2,:], 'g--')
		plt.plot(_r_t[1,i,3,:], 'r')
		plt.plot(r_t_true[1,i,3,:], 'r--')
		plt.legend(['tf', 'np'])
		plt.title('c %d i %d' % (1, i))
		plt.show()
	assert(approx_equal(_r_t, r_t_true, EPS))
	"""

	# ****************************************
	# Two random fixed params
	# ****************************************
	W_EE = 1.0
	h_LATE = 1.0
	fixed_params = {'W_EE':W_EE, 'h_LATE':h_LATE}
	system = V1Circuit(fixed_params, behavior1)
	# Test free/fixed parameter handling
	assert(system.fixed_params['W_EE'] == W_EE)
	assert(system.fixed_params['h_LATE'] == h_LATE)
	assert(len(system.fixed_params.keys()) == 2)
	assert(system.free_params == ['W_PE', 'W_SE', 'W_VE', \
									'b_E', 'b_P', 'b_S', 'b_V', \
									'h_FFE', 'h_FFP', \
									'h_LATP', 'h_LATS', 'h_LATV', \
									'h_RUNE', 'h_RUNP', 'h_RUNS', 'h_RUNV', \
									'tau', 'n', 's_0'])
	assert(system.z_labels == [r'$W_{PE}$', r'$W_{SE}$', r'$W_{VE}$', \
                      r'$b_{E}$', r'$b_{P}$', r'$b_{S}$', r'$b_{V}$', \
                      r'$h_{FF,E}$', r'$h_{FF,P}$', \
                      r'$h_{LAT,P}$', r'$h_{LAT,S}$', r'$h_{LAT,V}$', \
                      r'$h_{RUN,E}$', r'$h_{RUN,P}$', r'$h_{RUN,S}$', r'$h_{RUN,V}$', \
                      r'$\tau$', r'$n$', r'$s_0$'])
	assert(system.T_x_labels == [r'$d_{E,ss}$', r'$d_{P,ss}$', r'$d_{S,ss}$', r'$d_{V,ss}$', \
								r'$d_{E,ss}^2$', r'$d_{P,ss}^2$', r'$d_{S,ss}^2$', r'$d_{V,ss}^2$'])
	assert(system.D == 19)
	assert(system.num_suff_stats == 8)

	# Test density network output filtering
	Z = tf.placeholder(dtype=tf.float64, shape=(1, M, system.D))
	W, b, h_FF, h_LAT, h_RUN, tau, n, s_0, a, c_50 = system.filter_Z(Z)
	assert(a is None)
	assert(c_50 is None)

	_Z = np.random.normal(0.0, 1.0, (1, M, system.D))
	_W, _b, _h_FF, _h_LAT, _h_RUN, _tau, _n, _s_0 = sess.run([W, b, h_FF, h_LAT, h_RUN, tau, n, s_0], {Z:_Z})
	assert(_W.shape == (2,M,4,4))
	assert(_b.shape == (1,M,4,1))
	assert(_h_FF.shape == (1,M,4,1))
	assert(_h_LAT.shape == (1,M,4,1))
	assert(_h_RUN.shape == (1,M,4,1))
	assert(_tau.shape == (2,M,1,1))
	assert(_n.shape == (2,M,1,1))
	assert(_s_0.shape == (1,M,1,1))
	W_PSV = np.tile(np.array([[[-1.0,  -0.54, 0.0], \
							  [-1.01, -0.33, 0.0], \
							  [0.0,   0.0, -0.15], \
							  [-0.22, -0.77, 0.0]]]), [M, 1, 1]) 
	W_E = np.expand_dims(np.concatenate((W_EE*np.ones((M,1)), _Z[0,:,:3]), axis=1), 2)
	W_true = np.concatenate((W_E, W_PSV), axis=2)
	assert(approx_equal(_W[0], W_true, EPS))
	assert(approx_equal(_W[1], W_true, EPS))
	b_true = np.expand_dims(_Z[:,:,3:7], 3)
	assert(approx_equal(_b, b_true, EPS))
	h_FF_true = np.concatenate((np.expand_dims(_Z[:,:,7:9], 3), np.zeros((1, M, 2, 1))), axis=2)
	assert(approx_equal(_h_FF, h_FF_true, EPS))
	h_LAT_true = np.expand_dims(np.concatenate((h_LATE*np.ones((1,M,1)), _Z[:,:,9:12]), axis=2), 3)
	assert(approx_equal(_h_LAT, h_LAT_true, EPS))
	h_RUN_true = np.expand_dims(_Z[:,:,12:16], 3)
	assert(approx_equal(_h_RUN, h_RUN_true, EPS))
	tau_true = np.tile(np.expand_dims(_Z[:,:,16:17], 3), [2,1,1,1])
	assert(approx_equal(_tau, tau_true, EPS))
	n_true = np.tile(np.expand_dims(_Z[:,:,17:18], 3), [2,1,1,1])
	assert(approx_equal(_n, n_true, EPS))
	s_0_true = np.expand_dims(_Z[:,:,18:19], 3)
	assert(approx_equal(_s_0, s_0_true, EPS))


	# ****************************************
	# Five random fixed params
	# ****************************************
	b_V = 1.0
	h_LATP = 1.0
	h_RUNS = 1.0
	tau = 1.0
	n = 1.0
	fixed_params = {'b_V':b_V, 'h_LATP':h_LATP, 'h_RUNS':h_RUNS, 'tau':tau, 'n':n}
	system = V1Circuit(fixed_params, behavior1)
	# Test free/fixed parameter handling
	assert(system.fixed_params['b_V'] == b_V)
	assert(system.fixed_params['h_LATP'] == h_LATP)
	assert(system.fixed_params['h_RUNS'] == h_RUNS)
	assert(system.fixed_params['tau'] == tau)
	assert(system.fixed_params['n'] == n)
	assert(len(system.fixed_params.keys()) == 5)
	assert(system.free_params == ['W_EE', 'W_PE', 'W_SE', 'W_VE', \
									'b_E', 'b_P', 'b_S', \
									'h_FFE', 'h_FFP', \
									'h_LATE', 'h_LATS', 'h_LATV', \
									'h_RUNE', 'h_RUNP', 'h_RUNV', \
									's_0'])
	assert(system.z_labels == [r'$W_{EE}$', r'$W_{PE}$', r'$W_{SE}$', r'$W_{VE}$', \
                      r'$b_{E}$', r'$b_{P}$', r'$b_{S}$', \
                      r'$h_{FF,E}$', r'$h_{FF,P}$', \
                      r'$h_{LAT,E}$', r'$h_{LAT,S}$', r'$h_{LAT,V}$', \
                      r'$h_{RUN,E}$', r'$h_{RUN,P}$', r'$h_{RUN,V}$', \
                      r'$s_0$'])
	assert(system.T_x_labels == [r'$d_{E,ss}$', r'$d_{P,ss}$', r'$d_{S,ss}$', r'$d_{V,ss}$', \
								r'$d_{E,ss}^2$', r'$d_{P,ss}^2$', r'$d_{S,ss}^2$', r'$d_{V,ss}^2$'])
	assert(system.D == 16)
	assert(system.num_suff_stats == 8)

	# Test density network output filtering
	Z = tf.placeholder(dtype=tf.float64, shape=(1, M, system.D))
	W, b, h_FF, h_LAT, h_RUN, tau_tf, n_tf, s_0, a, c_50 = system.filter_Z(Z)
	assert(a is None)
	assert(c_50 is None)

	_Z = np.random.normal(0.0, 1.0, (1, M, system.D))
	_W, _b, _h_FF, _h_LAT, _h_RUN, _tau, _n, _s_0 = sess.run([W, b, h_FF, h_LAT, h_RUN, tau_tf, n_tf, s_0], {Z:_Z})
	assert(_W.shape == (2,M,4,4))
	assert(_b.shape == (1,M,4,1))
	assert(_h_FF.shape == (1,M,4,1))
	assert(_h_LAT.shape == (1,M,4,1))
	assert(_h_RUN.shape == (1,M,4,1))
	assert(_tau.shape == (2,M,1,1))
	assert(_n.shape == (2,M,1,1))
	assert(_s_0.shape == (1,M,1,1))
	W_PSV = np.tile(np.array([[[-1.0,  -0.54, 0.0], \
							  [-1.01, -0.33, 0.0], \
							  [0.0,   0.0, -0.15], \
							  [-0.22, -0.77, 0.0]]]), [M, 1, 1])
	W_E = np.expand_dims(_Z[0,:,:4], 2)
	W_true = np.concatenate((W_E, W_PSV), axis=2)
	assert(approx_equal(_W[0], W_true, EPS))
	assert(approx_equal(_W[1], W_true, EPS))
	b_true = np.expand_dims(np.concatenate((_Z[:,:,4:7], b_V*np.ones((1,M,1))), axis=2), 3)
	assert(approx_equal(_b, b_true, EPS))
	h_FF_true = np.concatenate((np.expand_dims(_Z[:,:,7:9], 3), np.zeros((1, M, 2, 1))), axis=2)
	assert(approx_equal(_h_FF, h_FF_true, EPS))
	h_LAT_true = np.expand_dims(np.concatenate((_Z[:,:,9:10], h_LATP*np.ones((1,M,1)), _Z[:,:,10:12]), axis=2), 3)
	assert(approx_equal(_h_LAT, h_LAT_true, EPS))
	h_RUN_true = np.expand_dims(np.concatenate((_Z[:,:,12:14], h_RUNS*np.ones((1,M,1)), _Z[:,:,14:15]), axis=2), 3)
	assert(approx_equal(_h_RUN, h_RUN_true, EPS))
	tau_true = np.tile(np.expand_dims(tau*np.ones((1,M,1)), 3), [2,1,1,1])
	assert(approx_equal(_tau, tau_true, EPS))
	n_true = np.tile(np.expand_dims(n*np.ones((1,M,1)), 3), [2,1,1,1])
	assert(approx_equal(_n, n_true, EPS))
	s_0_true = np.expand_dims(_Z[:,:,15:16], 3)
	assert(approx_equal(_s_0, s_0_true, EPS))

	print('V1Circuit passed.')
	return None



def test_LowRankRNN():
	n = 30
	with tf.Session() as sess:
		true_sys = low_rank_rnn()
		Z = tf.placeholder(dtype=DTYPE, shape=(1, n, None))

		mu1 = np.array([1.0, 1.0, 1.0]);
		Sigma1 = np.array([0.1, 0.1, 0.1]);
		behavior1 = {'type':'struct_chaos', 'means':mu1, 'variances':Sigma1};
		model_opts = {'rank':1, 'input_type':'spont'}

		# no fixed parameters
		sys = LowRankRNN({}, behavior1, model_opts=model_opts)
		assert(sys.name == 'LowRankRNN')
		assert(not sys.fixed_params) # empty dict evaluates to False
		assert(sys.all_params == ['g', 'Mm', 'Mn', 'Sm'])
		assert(sys.free_params == ['g', 'Mm', 'Mn', 'Sm'])
		assert(sys.z_labels == [r'$g$', r'$M_m$', r'$M_n$', r'$\Sigma_m$'])
		assert(sys.T_x_labels == [r'$\mu$', r'$\delta_{\infty}$', r'$\delta_0 - \delta_{\infty}$', \
                          r'$\mu^2$', r'$\delta_{\infty}^2$', r'$(\delta_0 - \delta_{\infty})^2$'])
		assert(sys.D == 4)
		assert(sys.num_suff_stats == 6)

		# Fix Mm to 1.0
		Mm = 1.0
		sys = LowRankRNN({'Mm':Mm}, behavior1, model_opts=model_opts)
		assert(sys.fixed_params['Mm'] == Mm) # empty dict evaluates to False
		assert(sys.all_params == ['g', 'Mm', 'Mn', 'Sm'])
		assert(sys.free_params == ['g', 'Mn', 'Sm'])
		assert(sys.z_labels == [r'$g$', r'$M_n$', r'$\Sigma_m$'])
		assert(sys.T_x_labels == [r'$\mu$', r'$\delta_{\infty}$', r'$\delta_0 - \delta_{\infty}$', \
                          r'$\mu^2$', r'$\delta_{\infty}^2$', r'$(\delta_0 - \delta_{\infty})^2$'])
		assert(sys.D == 3)
		assert(sys.num_suff_stats == 6)

		# Fix all parameters
		g = 1.0
		Mm = 1.0
		Mn = 1.0
		Sm = 1.0
		sys = LowRankRNN({'g':g, 'Mm':Mm, 'Mn':Mn, 'Sm':Sm}, behavior1, model_opts=model_opts)
		assert(sys.fixed_params['g'] == g)
		assert(sys.fixed_params['Mm'] == Mm)
		assert(sys.fixed_params['Mn'] == Mn)
		assert(sys.fixed_params['Sm'] == Sm)
		assert(sys.all_params == ['g', 'Mm', 'Mn', 'Sm'])
		assert(sys.free_params == [])
		assert(sys.z_labels == [])
		assert(sys.T_x_labels == [r'$\mu$', r'$\delta_{\infty}$', r'$\delta_0 - \delta_{\infty}$', \
                          r'$\mu^2$', r'$\delta_{\infty}^2$', r'$(\delta_0 - \delta_{\infty})^2$'])
		assert(sys.D == 0)
		assert(sys.num_suff_stats == 6)


		# test mu computation
		print('mu testing')
		sys = LowRankRNN({}, behavior1, model_opts=model_opts)
		mu1 = true_sys.compute_mu(behavior1)
		assert(approx_equal(sys.mu, mu1, EPS))

		mu2 = np.array([4.0, 4.0, 10.0]);
		Sigma2 = np.array([0.001, 1000, 1.0]);
		behavior2 = {'type':'struct_chaos', 'means':mu2, 'variances':Sigma2};

		mu3 = np.array([-4.0, -4.0, -10.0]);
		Sigma3 = np.array([0.001, 1000, 1.0]);
		behavior3 = {'type':'struct_chaos', 'means':mu3, 'variances':Sigma3};

		sys = LowRankRNN({}, behavior2)
		mu2 = true_sys.compute_mu(behavior2)
		assert(approx_equal(sys.mu, mu2, EPS))

		sys = LowRankRNN({}, behavior3)
		mu3 = true_sys.compute_mu(behavior3)
		assert(approx_equal(sys.mu, mu3, EPS))

		# test sufficient statistic computation
		sys = LowRankRNN({}, behavior1, model_opts=model_opts, solve_its=200, solve_eps=0.2)
		T_x = sys.compute_suff_stats(Z)
		g = np.abs(np.random.normal(1.0, 1.0, (1,n,1))) + 0.5
		Mm = np.abs(np.random.normal(1.0, 1.0, (1,n,1))) + 0.5
		Mn = np.abs(np.random.normal(1.0, 1.0, (1,n,1))) + 0.5
		Sm = np.abs(np.random.normal(1.0, 1.0, (1,n,1))) + 0.5
		_Z = np.concatenate((g, Mm, Mn, Sm), axis=2)
		_T_x_true = np.zeros((n,sys.num_suff_stats))
		for i in range(n):
			_T_x_true[i,:] = true_sys.compute_suff_stats(g[0,i,0], Mm[0,i,0], Mn[0,i,0], Sm[0,i,0])
		_T_x = sess.run(T_x, {Z:_Z})
		
		for i in range(3):
			plt.figure()
			plt.scatter(_T_x[0,:,i], _T_x_true[:,i])
			maxval = max([np.max(_T_x[0,:,i]), np.max(_T_x_true[:,i])])
			plt.plot([0, maxval], [0, maxval], 'k--')
			plt.show()

		# TODO add tests for all of the different parameterizations and behaviors (tasks)

	print('LowRankRNN passed.')
	return None


if __name__ == "__main__":
	#test_Linear2D()
	#test_V1Circuit()
	test_LowRankRNN()


