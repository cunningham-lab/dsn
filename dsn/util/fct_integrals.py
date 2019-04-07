# Francesca Mastrogiuseppe 2018

import numpy as np

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
#### Computing Gaussian integrals through Gauss-Hermite quadrature
# here phi(x) = tanh(x)

# Global variables for Gaussian quadrature


def get_quadrature(num_pts):
    gaussian_norm = 1 / np.sqrt(np.pi)
    gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(num_pts)
    gauss_points = gauss_points * np.sqrt(2)
    return gaussian_norm, gauss_points, gauss_weights


#### Single Gaussian intergrals


def Prim(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    integrand = np.log(np.cosh(mu + np.sqrt(delta0) * gauss_points))
    return gaussian_norm * np.dot(integrand, gauss_weights)


def Phi(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    integrand = np.tanh(mu + np.sqrt(delta0) * gauss_points)
    return gaussian_norm * np.dot(integrand, gauss_weights)


def Prime(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    integrand = 1 - (np.tanh(mu + np.sqrt(delta0) * gauss_points)) ** 2
    return gaussian_norm * np.dot(integrand, gauss_weights)


def Sec(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    integrand = (
        -2
        * np.tanh(mu + np.sqrt(delta0) * gauss_points)
        * (1 - (np.tanh(mu + np.sqrt(delta0) * gauss_points)) ** 2)
    )
    return gaussian_norm * np.dot(integrand, gauss_weights)


def Third(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    integrand = (
        -2
        * (3 * np.tanh(mu + np.sqrt(delta0) * gauss_points) ** 2 - 1)
        * (1 - (np.tanh(mu + np.sqrt(delta0) * gauss_points)) ** 2)
    )
    return gaussian_norm * np.dot(integrand, gauss_weights)


def PrimSq(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    integrand = np.log(np.cosh(mu + np.sqrt(delta0) * gauss_points))
    return gaussian_norm * np.dot(integrand ** 2, gauss_weights)


def PhiSq(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    integrand = np.tanh(mu + np.sqrt(delta0) * gauss_points)
    return gaussian_norm * np.dot(integrand ** 2, gauss_weights)


def PrimeSq(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    integrand = 1 - (np.tanh(mu + np.sqrt(delta0) * gauss_points)) ** 2
    return gaussian_norm * np.dot(integrand ** 2, gauss_weights)


def PhiPrime(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    integrand = np.tanh(mu + np.sqrt(delta0) * gauss_points) * (
        1 - (np.tanh(mu + np.sqrt(delta0) * gauss_points)) ** 2
    )
    return gaussian_norm * np.dot(integrand, gauss_weights)


def PrimPrime(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    integrand = (1 - (np.tanh(mu + np.sqrt(delta0) * gauss_points)) ** 2) * np.log(
        np.cosh(mu + np.sqrt(delta0) * gauss_points)
    )
    return gaussian_norm * np.dot(integrand, gauss_weights)


def PhiSec(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    integrand = (
        -2
        * (np.tanh(mu + np.sqrt(delta0) * gauss_points) ** 2)
        * (1 - (np.tanh(mu + np.sqrt(delta0) * gauss_points)) ** 2)
    )
    return gaussian_norm * np.dot(integrand, gauss_weights)


def PrimPhi(mu, delta0, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    integrand = np.log(np.cosh(mu + np.sqrt(delta0) * gauss_points)) * np.tanh(
        mu + np.sqrt(delta0) * gauss_points
    )
    return gaussian_norm * np.dot(integrand, gauss_weights)


#### Nested Gaussian intergrals


def InnerPrimPrim(
    z, mu, delta0, deltainf, num_pts=200
):  # Performs the internal integral
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    integrand = np.log(
        np.cosh(mu + np.sqrt(delta0 - deltainf) * gauss_points + np.sqrt(deltainf) * z)
    )
    return gaussian_norm * np.dot(integrand, gauss_weights)


def IntPrimPrim(mu, delta0, deltainf, num_pts=200):  # Performs the external integral
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    integrand = InnerPrimPrim(gauss_points, mu, delta0, deltainf, num_pts)
    return gaussian_norm * np.dot(integrand ** 2, gauss_weights)


InnerPrimPrim = np.vectorize(InnerPrimPrim)


def InnerPhiPhi(z, mu, delta0, deltainf, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    integrand = np.tanh(
        mu + np.sqrt(delta0 - deltainf) * gauss_points + np.sqrt(deltainf) * z
    )
    return gaussian_norm * np.dot(integrand, gauss_weights)


def IntPhiPhi(mu, delta0, deltainf, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    integrand = InnerPhiPhi(gauss_points, mu, delta0, deltainf, num_pts)
    return gaussian_norm * np.dot(integrand ** 2, gauss_weights)


InnerPhiPhi = np.vectorize(InnerPhiPhi)


def InnerPrimePrime(z, mu, delta0, deltainf, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    integrand = (
        1
        - np.tanh(
            mu + np.sqrt(delta0 - deltainf) * gauss_points + np.sqrt(deltainf) * z
        )
        ** 2
    )
    return gaussian_norm * np.dot(integrand, gauss_weights)


def IntPrimePrime(mu, delta0, deltainf, num_pts=200):
    gaussian_norm, gauss_points, gauss_weights = get_quadrature(num_pts)
    integrand = InnerPrimePrime(gauss_points, mu, delta0, deltainf, num_pts)
    return gaussian_norm * np.dot(integrand ** 2, gauss_weights)


InnerPrimePrime = np.vectorize(InnerPrimePrime)
