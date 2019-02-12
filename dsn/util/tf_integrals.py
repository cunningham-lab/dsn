import numpy as np
import tensorflow as tf

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Computing Gaussian integrals through Gauss-Hermite quadrature
# here phi(x) = tanh(x)

# Global variables for Gaussian quadrature
def get_quadrature(num_pts, double=False):
    gauss_norm = 1 / np.sqrt(np.pi)
    gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(num_pts)
    gauss_points = np.expand_dims(gauss_points * np.sqrt(2), 0)

    if double:
        gauss_points_inner = np.expand_dims(gauss_points, 1)
        gauss_points_outer = np.expand_dims(gauss_points, 2)
        return (
            gauss_norm,
            gauss_points,
            gauss_weights,
            gauss_points_inner,
            gauss_points_outer,
        )
    else:
        return gauss_norm, gauss_points, gauss_weights


#### Single Gaussian intergrals


def Prim(mu, delta0, num_pts=200):
    gauss_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = tf.expand_dims(mu, 1)
    delta0 = tf.expand_dims(delta0, 1)
    integrand = tf.log(tf.math.cosh(mu + tf.sqrt(delta0) * gauss_points))
    return gauss_norm * tf.tensordot(integrand, gauss_weights, [[1], [0]])


def Phi(mu, delta0, num_pts=200):
    gauss_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = tf.expand_dims(mu, 1)
    delta0 = tf.expand_dims(delta0, 1)
    integrand = tf.tanh(mu + tf.sqrt(delta0) * gauss_points)
    return gauss_norm * tf.tensordot(integrand, gauss_weights, [[1], [0]])


def Prime(mu, delta0, num_pts=200):
    gauss_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = tf.expand_dims(mu, 1)
    delta0 = tf.expand_dims(delta0, 1)
    integrand = 1 - tf.square(tf.tanh(mu + tf.sqrt(delta0) * gauss_points))
    return gauss_norm * tf.tensordot(integrand, gauss_weights, [[1], [0]])


def Sec(mu, delta0, num_pts=200):
    gauss_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = tf.expand_dims(mu, 1)
    delta0 = tf.expand_dims(delta0, 1)
    integrand = (
        -2
        * tf.tanh(mu + tf.sqrt(delta0) * gauss_points)
        * (1 - (tf.tanh(mu + tf.sqrt(delta0) * gauss_points)) ** 2)
    )
    return gauss_norm * tf.tensordot(integrand, gauss_weights, [[1], [0]])


def Third(mu, delta0, num_pts=200):
    gauss_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = tf.expand_dims(mu, 1)
    delta0 = tf.expand_dims(delta0, 1)
    integrand = (
        -2
        * (3 * tf.tanh(mu + tf.sqrt(delta0) * gauss_points) ** 2 - 1)
        * (1 - (tf.tanh(mu + tf.sqrt(delta0) * gauss_points)) ** 2)
    )
    return gauss_norm * tf.tensordot(integrand, gauss_weights, [[1], [0]])


#


def PrimSq(mu, delta0, num_pts=200):
    gauss_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = tf.expand_dims(mu, 1)
    delta0 = tf.expand_dims(delta0, 1)
    integrand = tf.log(tf.math.cosh(mu + tf.sqrt(delta0) * gauss_points))
    return gauss_norm * tf.tensordot(integrand ** 2, gauss_weights, [[1], [0]])


def PhiSq(mu, delta0, num_pts=200):
    gauss_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = tf.expand_dims(mu, 1)
    delta0 = tf.expand_dims(delta0, 1)
    integrand = tf.tanh(mu + tf.sqrt(delta0) * gauss_points)
    return gauss_norm * tf.tensordot(tf.square(integrand), gauss_weights, [[1], [0]])


def PrimeSq(mu, delta0, num_pts=200):
    gauss_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = tf.expand_dims(mu, 1)
    delta0 = tf.expand_dims(delta0, 1)
    integrand = 1 - (tf.tanh(mu + tf.sqrt(delta0) * gauss_points)) ** 2
    return gauss_norm * tf.tensordot(integrand ** 2, gauss_weights, [[1], [0]])


def PhiPrime(mu, delta0, num_pts=200):
    gauss_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = tf.expand_dims(mu, 1)
    delta0 = tf.expand_dims(delta0, 1)
    integrand = tf.tanh(mu + tf.sqrt(delta0) * gauss_points) * (
        1 - (tf.tanh(mu + tf.sqrt(delta0) * gauss_points)) ** 2
    )
    return gauss_norm * tf.tensordot(integrand, gauss_weights, [[1], [0]])


def PrimPrime(mu, delta0, num_pts=200):
    gauss_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = tf.expand_dims(mu, 1)
    delta0 = tf.expand_dims(delta0, 1)
    integrand = (1 - (tf.tanh(mu + tf.sqrt(delta0) * gauss_points)) ** 2) * tf.log(
        tf.math.cosh(mu + tf.sqrt(delta0) * gauss_points)
    )
    return gauss_norm * tf.tensordot(integrand, gauss_weights, [[1], [0]])


def PhiSec(mu, delta0, num_pts=200):
    gauss_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = tf.expand_dims(mu, 1)
    delta0 = tf.expand_dims(delta0, 1)
    integrand = (
        -2
        * (tf.tanh(mu + tf.sqrt(delta0) * gauss_points) ** 2)
        * (1 - (tf.tanh(mu + tf.sqrt(delta0) * gauss_points)) ** 2)
    )
    return gauss_norm * tf.tensordot(integrand, gauss_weights, [[1], [0]])


def PrimPhi(mu, delta0, num_pts=200):
    gauss_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = tf.expand_dims(mu, 1)
    delta0 = tf.expand_dims(delta0, 1)
    integrand = tf.log(tf.math.cosh(mu + tf.sqrt(delta0) * gauss_points)) * tf.tanh(
        mu + tf.sqrt(delta0) * gauss_points
    )
    return gauss_norm * tf.tensordot(integrand, gauss_weights, [[1], [0]])


#### Nested Gaussian intergrals


def IntPrimPrim(mu, delta0, deltainf, num_pts=200):  # Performs the external integral
    gauss_norm, gauss_points, gauss_weights, gauss_points_inner, gauss_points_outer = get_quadrature(
        num_pts, double=True
    )
    mu = tf.expand_dims(tf.expand_dims(mu, 1), 2)
    delta0 = tf.expand_dims(tf.expand_dims(delta0, 1), 2)
    deltainf = tf.expand_dims(tf.expand_dims(deltainf, 1), 2)
    inner_integrand = tf.log(
        tf.math.cosh(
            mu
            + tf.sqrt(delta0 - deltainf) * gauss_points_inner
            + tf.sqrt(deltainf) * gauss_points_outer
        )
    )
    outer_integrand = gauss_norm * tf.tensordot(
        inner_integrand, gauss_weights, [[2], [0]]
    )
    return gauss_norm * tf.tensordot(outer_integrand ** 2, gauss_weights, [[1], [0]])


def IntPhiPhi(mu, delta0, deltainf, num_pts=200):
    gauss_norm, gauss_points, gauss_weights, gauss_points_inner, gauss_points_outer = get_quadrature(
        num_pts, double=True
    )
    mu = tf.expand_dims(tf.expand_dims(mu, 1), 2)
    delta0 = tf.expand_dims(tf.expand_dims(delta0, 1), 2)
    deltainf = tf.expand_dims(tf.expand_dims(deltainf, 1), 2)
    inner_integrand = tf.tanh(
        mu
        + tf.sqrt(delta0 - deltainf) * gauss_points_inner
        + tf.sqrt(deltainf) * gauss_points_outer
    )
    outer_integrand = gauss_norm * tf.tensordot(
        inner_integrand, gauss_weights, [[2], [0]]
    )
    return gauss_norm * tf.tensordot(outer_integrand ** 2, gauss_weights, [[1], [0]])


def IntPrimePrime(mu, delta0, deltainf, num_pts=200):
    gauss_norm, gauss_points, gauss_weights, gauss_points_inner, gauss_points_outer = get_quadrature(
        num_pts, double=True
    )
    mu = tf.expand_dims(tf.expand_dims(mu, 1), 2)
    delta0 = tf.expand_dims(tf.expand_dims(delta0, 1), 2)
    deltainf = tf.expand_dims(tf.expand_dims(deltainf, 1), 2)
    inner_integrand = (
        1
        - tf.tanh(
            mu
            + tf.sqrt(delta0 - deltainf) * gauss_points_inner
            + tf.sqrt(deltainf) * gauss_points_outer
        )
        ** 2
    )
    outer_integrand = gauss_norm * tf.tensordot(
        inner_integrand, gauss_weights, [[2], [0]]
    )
    return gauss_norm * tf.tensordot(outer_integrand ** 2, gauss_weights, [[1], [0]])
