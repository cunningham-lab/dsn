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

MAXVAL = 150.0

def bounded_langevin_dyn(f, x0, eps, num_its, non_neg, db=False):
    """Tensorflow langevin dynamics

        # Arguments:
            f (function): maps tf.tensor (M,d) MF coeff to consist eq.
            x0 (tf.tensor): (M,d) initial conditions.
            eps (float): langevin dyanmics step size.
            num_its (int): number of iterations.
            non_neg (list): True if dimension is nonnegative. 

        # Returns
            x_i (tf.tensor): (M,d) consistency equation solution
        """
    d = len(non_neg)
    x_i = x0
    if db:
        xs = [x0]
    for i in range(num_its):
        f_x = f(x_i)
        x_next = []
        for j in range(d):
            if (non_neg[j]):
                x_next.append(
                    tf.clip_by_value(
                        (1.0 - eps) * x_i[:, j] + eps * f_x[:, j], 0.0, MAXVAL
                    )
                )
            else:
                x_next.append(
                    tf.clip_by_value(
                        (1.0 - eps) * x_i[:, j] + eps * f_x[:, j], -MAXVAL, MAXVAL
                    )
                )
        x_i = tf.stack(x_next, axis=1)
        if db:
            xs.append(x_i)
    
    if db:
        return x_i, tf.stack(xs, axis=2)
    else:
        return x_i

def bounded_langevin_dyn_np(f, x0, eps, num_its, non_neg, db=False):
    """Tensorflow langevin dynamics

        # Arguments:
            f (function): maps np.arrays (M,d) MF coeff to consist eqs.
            x0 (np.array): (M,d) initial conditions.
            eps (float): langevin dyanmics step size.
            num_its (int): number of iterations.
            non_neg (list): True if dimension is nonnegative. 

        # Returns
            x_i (np.array): (M,d) consistency equation solution
        """
    d = len(non_neg)
    x_i = x0
    if db:
        xs = [x0]
    for i in range(num_its):
        f_x = f(x_i)
        x_next = []
        for j in range(d):
            if (non_neg[j]):
                x_next.append(
                    np.clip(
                        (1.0 - eps) * x_i[:, j] + eps * f_x[:, j], 0.0, MAXVAL
                    )
                )
            else:
                x_next.append(
                    np.clip(
                        (1.0 - eps) * x_i[:, j] + eps * f_x[:, j], -MAXVAL, MAXVAL
                    )
                )
        x_i = np.stack(x_next, axis=1)
        if db:
            xs.append(x_i)
    
    if db:
        return x_i, np.stack(xs, axis=2)
    else:
        return x_i


