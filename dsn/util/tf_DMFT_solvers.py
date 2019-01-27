import numpy as np
import tensorflow as tf
import dsn.util.tf_integrals as tfi
from dsn.util.tf_langevin import langevin_dyn, langevin_dyn_spont_chaos

def spont_static_solve(mu_init, delta_0_init, g, Mm, Mn, Sm, num_its, eps):

	# convergence equations used for langevin-like dynamimcs solver 
	def f(x):
		mu = x[:,0]
		delta_0 = x[:,1]

		Phi = tfi.Phi(mu, delta_0)
		PhiSq = tfi.PhiSq(mu, delta_0)

		F = Mm*Mn*Phi
		H = (g**2)*PhiSq + (Sm**2)*(Mn**2)*Phi**2
		return tf.stack([F, H], axis=1)

	x_init = tf.stack([mu_init, delta_0_init], axis=1)
	xs_end = langevin_dyn(f, x_init, eps, num_its)
	mu = xs_end[:,0]
	delta_0 = xs_end[:,1]
	return mu, delta_0

def spont_chaotic_solve(mu_init, delta_0_init, delta_inf_init, g, Mm, Mn, Sm, num_its, eps, db=False):

	# convergence equations used for langevin-like dynamimcs solver 
	def f(x):
		mu = x[:,0]
		delta_0 = x[:,1]
		delta_inf = x[:,2]

		Phi = tfi.Phi(mu, delta_0)
		PrimSq = tfi.PrimSq(mu, delta_0)
		IntPrimPrim = tfi.IntPrimPrim(mu, delta_0, delta_inf)
		IntPhiPhi = tfi.IntPhiPhi(mu, delta_0, delta_inf)

		F = Mm*Mn*Phi
		H_squared = delta_inf**2 + 2*((g**2)*(PrimSq - IntPrimPrim) + (Mn**2)*(Sm**2)*(Phi**2)*(delta_0 - delta_inf))
		H = tf.sqrt(tf.nn.relu(H_squared))
		G = (g**2)*IntPhiPhi + (Mn**2)*(Sm**2)*(Phi**2)
		return tf.stack([F, H, G], axis=1)

	x_init = tf.stack([mu_init, delta_0_init, delta_inf_init], axis=1)
	if db:
		xs_end, xs = langevin_dyn_spont_chaos(f, x_init, eps, num_its, db=db)
	else:
		xs_end = langevin_dyn_spont_chaos(f, x_init, eps, num_its, db=db)

	mu = xs_end[:,0]
	delta_0 = xs_end[:,1]
	delta_inf = xs_end[:,2]

	if db:
		return mu, delta_0, delta_inf, xs
	else:
		return mu, delta_0, delta_inf














