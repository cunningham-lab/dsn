import tensorflow as tf
import numpy as np
import scipy
from tf_util.stat_util import approx_equal
from dsn.util.systems import system
import matplotlib.pyplot as plt

DTYPE = tf.float64
EPS = 1e-16

# Testing the functionality of the base system class
def test_system():
    fixed_params = {}
    behavior = {'type':'oscillation', 'f':2.0}
    # base methods called in init are not implmemented for system
    caught_except = False
    try:
        sys = system(fixed_params, behavior)
    except NotImplementedError:
        caught_except = True
    assert(caught_except)

    class TestSystem(system):
        def get_all_sys_params(self,):
            all_params = [
                "a",
                "b",
                "c",
            ]
            all_param_labels = {
                "a": ["a"],
                "b": ["b"],
                "c": ["c"],
            }
            return all_params, all_param_labels

        def get_T_x_labels(self,):
            if (self.behavior['type'] == 'oscillation'):
                T_x_labels = [
                              r"$f$",
                              r"$f^2$",
                              ]
            else:
                raise NotImplementedError
            return T_x_labels

        def compute_mu(self,):
            if (self.behavior['type'] == 'oscillation'):
                f = self.behavior['f']
                mu = np.array([f, np.square(f)])
            else:
                raise NotImplementedError
            return mu

    fixed_params = {}
    behavior = {'type':'oscillation', 'f':2.0}
    test_sys = TestSystem(fixed_params, behavior)
    assert(len(test_sys.all_params) == 3)
    assert(len(test_sys.fixed_params.keys()) == 0)
    assert(test_sys.D == 3)
    assert(len(test_sys.free_params) == test_sys.D)
    assert(len(test_sys.z_labels) == test_sys.D)
    assert(test_sys.num_suff_stats == 2)
    assert(len(test_sys.T_x_labels) == test_sys.num_suff_stats)
    assert(approx_equal(test_sys.mu, np.array([2.0, 4.0]), EPS))
    assert(approx_equal(test_sys.density_network_init_mu, np.zeros((test_sys.D,)), EPS))
    assert(test_sys.density_network_bounds is None)

    fixed_params = {'a':1}
    test_sys = TestSystem(fixed_params, behavior)
    assert(len(test_sys.all_params) == 3)
    assert(len(test_sys.fixed_params.keys()) == 1)
    assert(test_sys.D == 2)
    assert(len(test_sys.free_params) == test_sys.D)
    assert(len(test_sys.z_labels) == test_sys.D)
    assert(test_sys.fixed_params['a'] == 1)
    assert(test_sys.free_params[0] == 'b')
    assert(test_sys.free_params[1] == 'c')
    assert(test_sys.z_labels[0] == 'b')
    assert(test_sys.z_labels[1] == 'c')

    fixed_params = {'b':2, 'c':3}
    test_sys = TestSystem(fixed_params, behavior)
    assert(len(test_sys.all_params) == 3)
    assert(len(test_sys.fixed_params.keys()) == 2)
    assert(test_sys.D == 1)
    assert(len(test_sys.free_params) == test_sys.D)
    assert(len(test_sys.z_labels) == test_sys.D)
    assert(test_sys.fixed_params['b'] == 2)
    assert(test_sys.fixed_params['c'] == 3)
    assert(test_sys.free_params[0] == 'a')
    assert(test_sys.z_labels[0] == 'a')

    fixed_params = {'a':1, 'b':2, 'c':3}
    test_sys = TestSystem(fixed_params, behavior)
    assert(len(test_sys.all_params) == 3)
    assert(len(test_sys.fixed_params.keys()) == 3)
    assert(test_sys.D == 0)
    assert(len(test_sys.free_params) == test_sys.D)
    assert(len(test_sys.z_labels) == test_sys.D)
    assert(test_sys.fixed_params['a'] == 1)
    assert(test_sys.fixed_params['b'] == 2)
    assert(test_sys.fixed_params['c'] == 3)

    return None


if __name__ == "__main__":
    test_system()
