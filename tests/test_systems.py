import tensorflow as tf
import numpy as np
from tf_util.stat_util import approx_equal
from dsn.util.systems import system, Linear2D

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
	                          r'real($\lambda_1$)^2', r'imag($\lambda_1$)^2'])
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
	                          r'real($\lambda_1$)^2', r'imag($\lambda_1$)^2'])
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
	                           r'real($\lambda_1$)^2', r'imag($\lambda_1$)^2'])
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
	                           r'real($\lambda_1$)^2', r'imag($\lambda_1$)^2'])
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

if __name__ == "__main__":
	test_Linear2D()
