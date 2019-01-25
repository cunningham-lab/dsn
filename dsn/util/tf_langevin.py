import numpy as np
import tensorflow as tf

def langevin_dyn(f, x0, eps, num_its):
	x_i = x0
	xs = [x0]
	for i in range(num_its):
		x_i = (1.0-eps)*x_i + eps*f(x_i)
		xs.append(x_i)
	xs = tf.stack(xs, axis=2)
	return xs