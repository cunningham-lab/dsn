import numpy as np
import tensorflow as tf


def langevin_dyn(f, x0, eps, num_its):
    x_i = x0
    for i in range(num_its):
        x_i = (1.0 - eps) * x_i + eps * f(x_i)
    return x_i


MAXVAL = 150
EPS = 1e-4


def langevin_dyn_spont_chaos(f, x0, eps, num_its, db=False):
    x_i = x0
    if db:
        xs = [x0]
    for i in range(num_its):
        f_x = f(x_i)
        x_1 = tf.clip_by_value(
            (1.0 - eps) * x_i[:, 0] + eps * f_x[:, 0], -MAXVAL, MAXVAL
        )
        # delta0 should not be negative
        x_2 = tf.clip_by_value((1.0 - eps) * x_i[:, 1] + eps * f_x[:, 1], 0.0, MAXVAL)
        # deltainf should not be negative or greater than delta0
        x_3 = tf.clip_by_value(
            (1.0 - eps) * x_i[:, 2] + eps * f_x[:, 2], 0.0, x_2 - EPS
        )
        x_i = tf.stack([x_1, x_2, x_3], axis=1)
        if db:
            xs.append(x_i)
    if db:
        return x_i, tf.stack(xs, axis=2)
    else:
        return x_i
