# Copyright 2019 Sean Bittner, Columbia University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
import numpy as np
import tensorflow as tf


def langevin_dyn(f, x0, eps, num_its, db=False):
    x_i = x0
    for i in range(num_its):
        x_i = (1.0 - eps) * x_i + eps * f(x_i)
    return x_i


MAXVAL = 150
EPS = 1e-16

def langevin_dyn_rank1_spont_static(f, x0, eps, num_its, db=False):
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
        x_i = tf.stack([x_1, x_2], axis=1)
        if db:
            xs.append(x_i)
    
    if db:
        return x_i, tf.stack(xs, axis=2)
    else:
        return x_i


def langevin_dyn_rank1_spont_chaos(f, x0, eps, num_its, db=False):
    """ Documentation for this function
    """
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

def langevin_dyn_rank1_input_chaos(f, x0, eps, num_its, db=False):
    x_i = x0
    if db:
        xs = [x0]
    for i in range(num_its):
        f_x = f(x_i)
        x_1 = tf.clip_by_value(
            (1.0 - eps) * x_i[:, 0] + eps * f_x[:, 0], -MAXVAL, MAXVAL
        )
        x_2 = tf.clip_by_value(
            (1.0 - eps) * x_i[:, 1] + eps * f_x[:, 1], -MAXVAL, MAXVAL
        )
        # delta0 should not be negative
        x_3 = tf.clip_by_value((1.0 - eps) * x_i[:, 2] + eps * f_x[:, 2], 0.0, MAXVAL)
        # deltainf should not be negative or greater than delta0
        x_4 = tf.clip_by_value(
            (1.0 - eps) * x_i[:, 3] + eps * f_x[:, 3], 0.0, x_3 - EPS
        )
        x_i = tf.stack([x_1, x_2, x_3, x_4], axis=1)
        if db:
            xs.append(x_i)
    if db:
        return x_i, tf.stack(xs, axis=2)
    else:
        return x_i

# The functions above and below are the same.  Kappas should be bounded by
# max abs val, while delta0 and deltainf need their nonneg and
# del0 >= delinf bounds.

def langevin_dyn_rank2_CDD_chaos(f, x0, eps, num_its, db=False):
    x_i = x0
    if db:
        xs = [x0]
    for i in range(num_its):
        f_x = f(x_i)
        x_1 = tf.clip_by_value(
            (1.0 - eps) * x_i[:, 0] + eps * f_x[:, 0], -MAXVAL, MAXVAL
        )
        x_2 = tf.clip_by_value(
            (1.0 - eps) * x_i[:, 1] + eps * f_x[:, 1], -MAXVAL, MAXVAL
        )
        # delta0 should not be negative
        x_3 = tf.clip_by_value((1.0 - eps) * x_i[:, 2] + eps * f_x[:, 2], 0.0, MAXVAL)
        # deltainf should not be negative or greater than delta0
        x_4 = tf.clip_by_value(
            (1.0 - eps) * x_i[:, 3] + eps * f_x[:, 3], 0.0, x_3 - EPS
        )
        x_i = tf.stack([x_1, x_2, x_3, x_4], axis=1)
        if db:
            xs.append(x_i)
    if db:
        return x_i, tf.stack(xs, axis=2)
    else:
        return x_i
