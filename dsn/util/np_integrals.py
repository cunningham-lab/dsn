# Francesca Mastrogiuseppe 2018
# more generally vectorized as in tf code for warm-starting

import numpy as np

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
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = np.expand_dims(mu, 1)
    delta0 = np.expand_dims(delta0, 1)
    integrand = np.log(np.cosh(mu + np.sqrt(delta0) * gauss_points))
    return gaussian_norm * np.dot(integrand, gauss_weights)


def Phi(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = np.expand_dims(mu, 1)
    delta0 = np.expand_dims(delta0, 1)
    integrand = np.tanh(mu + np.sqrt(delta0) * gauss_points)
    return gaussian_norm * np.dot(integrand, gauss_weights)


def Prime(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = np.expand_dims(mu, 1)
    delta0 = np.expand_dims(delta0, 1)
    integrand = 1 - (np.tanh(mu + np.sqrt(delta0) * gauss_points)) ** 2
    return gaussian_norm * np.dot(integrand, gauss_weights)


def Sec(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = np.expand_dims(mu, 1)
    delta0 = np.expand_dims(delta0, 1)
    integrand = (
        -2
        * np.tanh(mu + np.sqrt(delta0) * gauss_points)
        * (1 - (np.tanh(mu + np.sqrt(delta0) * gauss_points)) ** 2)
    )
    return gaussian_norm * np.dot(integrand, gauss_weights)


def Third(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = np.expand_dims(mu, 1)
    delta0 = np.expand_dims(delta0, 1)
    integrand = (
        -2
        * (3 * np.tanh(mu + np.sqrt(delta0) * gauss_points) ** 2 - 1)
        * (1 - (np.tanh(mu + np.sqrt(delta0) * gauss_points)) ** 2)
    )
    return gaussian_norm * np.dot(integrand, gauss_weights)


def PrimSq(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = np.expand_dims(mu, 1)
    delta0 = np.expand_dims(delta0, 1)
    integrand = np.log(np.cosh(mu + np.sqrt(delta0) * gauss_points))
    return gaussian_norm * np.dot(integrand ** 2, gauss_weights)


def PhiSq(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = np.expand_dims(mu, 1)
    delta0 = np.expand_dims(delta0, 1)
    integrand = np.tanh(mu + np.sqrt(delta0) * gauss_points)
    return gaussian_norm * np.dot(integrand ** 2, gauss_weights)


def PrimeSq(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    integrand = 1 - (np.tanh(mu + np.sqrt(delta0) * gauss_points)) ** 2
    return gaussian_norm * np.dot(integrand ** 2, gauss_weights)


def PhiPrime(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = np.expand_dims(mu, 1)
    delta0 = np.expand_dims(delta0, 1)
    integrand = np.tanh(mu + np.sqrt(delta0) * gauss_points) * (
        1 - (np.tanh(mu + np.sqrt(delta0) * gauss_points)) ** 2
    )
    return gaussian_norm * np.dot(integrand, gauss_weights)


def PrimPrime(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = np.expand_dims(mu, 1)
    delta0 = np.expand_dims(delta0, 1)
    integrand = (1 - (np.tanh(mu + np.sqrt(delta0) * gauss_points)) ** 2) * np.log(
        np.cosh(mu + np.sqrt(delta0) * gauss_points)
    )
    return gaussian_norm * np.dot(integrand, gauss_weights)


def PhiSec(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = np.expand_dims(mu, 1)
    delta0 = np.expand_dims(delta0, 1)
    integrand = (
        -2
        * (np.tanh(mu + np.sqrt(delta0) * gauss_points) ** 2)
        * (1 - (np.tanh(mu + np.sqrt(delta0) * gauss_points)) ** 2)
    )
    return gaussian_norm * np.dot(integrand, gauss_weights)


def PrimPhi(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    mu = np.expand_dims(mu, 1)
    delta0 = np.expand_dims(delta0, 1)
    integrand = np.log(np.cosh(mu + np.sqrt(delta0) * gauss_points)) * np.tanh(
        mu + np.sqrt(delta0) * gauss_points
    )
    return gaussian_norm * np.dot(integrand, gauss_weights)


#### Nested Gaussian intergrals


def IntPrimPrim(mu, delta0, deltainf, num_pts=200):  # Performs the external integral
    gauss_norm, gauss_points, gauss_weights, gauss_points_inner, gauss_points_outer = get_quadrature(
        num_pts, double=True
    )
    mu = np.expand_dims(np.expand_dims(mu, 1), 2)
    delta0 = np.expand_dims(np.expand_dims(delta0, 1), 2)
    deltainf = np.expand_dims(np.expand_dims(deltainf, 1), 2)
    inner_integrand = np.log(
        np.cosh(
            mu
            + np.sqrt(delta0 - deltainf) * gauss_points_inner
            + np.sqrt(deltainf) * gauss_points_outer
        )
    )
    outer_integrand = gauss_norm * np.dot(inner_integrand, gauss_weights)
    return gauss_norm * np.dot(outer_integrand ** 2, gauss_weights)


def IntPhiPhi(mu, delta0, deltainf, num_pts=200):
    gauss_norm, gauss_points, gauss_weights, gauss_points_inner, gauss_points_outer = get_quadrature(
        num_pts, double=True
    )
    mu = np.expand_dims(np.expand_dims(mu, 1), 2)
    delta0 = np.expand_dims(np.expand_dims(delta0, 1), 2)
    deltainf = np.expand_dims(np.expand_dims(deltainf, 1), 2)
    inner_integrand = np.tanh(
        mu
        + np.sqrt(delta0 - deltainf) * gauss_points_inner
        + np.sqrt(deltainf) * gauss_points_outer
    )
    outer_integrand = gauss_norm * np.dot(inner_integrand, gauss_weights)
    return gauss_norm * np.dot(outer_integrand ** 2, gauss_weights)


def IntPrimePrime(mu, delta0, deltainf, num_pts=200):
    gauss_norm, gauss_points, gauss_weights, gauss_points_inner, gauss_points_outer = get_quadrature(
        num_pts, double=True
    )
    mu = np.expand_dims(tf.expand_dims(mu, 1), 2)
    delta0 = np.expand_dims(np.expand_dims(delta0, 1), 2)
    deltainf = np.expand_dims(np.expand_dims(deltainf, 1), 2)
    inner_integrand = (
        1
        - np.tanh(
            mu
            + np.sqrt(delta0 - deltainf) * gauss_points_inner
            + np.sqrt(deltainf) * gauss_points_outer
        )
        ** 2
    )
    outer_integrand = gauss_norm * np.dot(inner_integrand, gauss_weights)
    return gauss_norm * np.dot(outer_integrand ** 2, gauss_weights)
