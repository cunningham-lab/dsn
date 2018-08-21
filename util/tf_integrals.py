import numpy as np
import tensorflow as tf

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### Computing Gaussian integrals through Gauss-Hermite quadrature
# here phi(x) = tanh(x)

# Global variables for Gaussian quadrature

gaussian_norm = (1/np.sqrt(np.pi))
gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(200)
gauss_points = np.expand_dims(np.expand_dims(gauss_points*np.sqrt(2), 0), 0);


#### Single Gaussian intergrals

def Prim (mu, delta0):
    integrand = np.log(np.cosh(mu+np.sqrt(delta0)*gauss_points))
    return gaussian_norm * np.dot (integrand,gauss_weights)

def Phi (mu, delta0):
    mu = tf.expand_dims(mu, 2);
    delta0 = tf.expand_dims(delta0, 2);
    integrand = tf.tanh(mu+tf.sqrt(delta0)*gauss_points)
    return gaussian_norm * tf.tensordot(integrand,gauss_weights, [2,0]);

def Prime (mu, delta0):
    mu = tf.expand_dims(mu, 2);
    delta0 = tf.expand_dims(delta0, 2);
    integrand = 1 - tf.square(tf.tanh(mu+tf.sqrt(delta0)*gauss_points));
    return gaussian_norm * tf.tensordot(integrand,gauss_weights,[2,0]);

def Sec (mu, delta0):
    integrand = - 2 * np.tanh(mu+np.sqrt(delta0)*gauss_points) * (1-(np.tanh(mu+np.sqrt(delta0)*gauss_points))**2)
    return gaussian_norm * np.dot (integrand,gauss_weights)

def Third (mu, delta0):
    integrand = - 2 * (3*np.tanh(mu+np.sqrt(delta0)*gauss_points)**2 -1) * (1-(np.tanh(mu+np.sqrt(delta0)*gauss_points))**2)
    return gaussian_norm * np.dot (integrand,gauss_weights)

#

def PrimSq (mu, delta0):
    integrand = np.log(np.cosh(mu+np.sqrt(delta0)*gauss_points))
    return gaussian_norm * np.dot (integrand**2,gauss_weights)

def PhiSq (mu, delta0):
    mu = tf.expand_dims(mu, 2);
    delta0 = tf.expand_dims(delta0, 2);
    integrand = tf.tanh(mu+tf.sqrt(delta0)*gauss_points)
    return gaussian_norm * tf.tensordot(tf.square(integrand), gauss_weights, [2,0]);

def PrimeSq (mu, delta0):
    integrand = 1 - (np.tanh(mu+np.sqrt(delta0)*gauss_points))**2
    return gaussian_norm * np.dot (integrand**2,gauss_weights)

#

def PhiPrime (mu, delta0):
    integrand = np.tanh(mu+np.sqrt(delta0)*gauss_points) * (1-(np.tanh(mu+np.sqrt(delta0)*gauss_points))**2)
    return gaussian_norm * np.dot (integrand,gauss_weights)

def PrimPrime (mu, delta0):
    integrand = (1-(np.tanh(mu+np.sqrt(delta0)*gauss_points))**2) * np.log(np.cosh(mu+np.sqrt(delta0)*gauss_points))
    return gaussian_norm * np.dot (integrand,gauss_weights)

def PhiSec (mu, delta0):
    integrand = - 2 * (np.tanh(mu+np.sqrt(delta0)*gauss_points)**2 ) * (1-(np.tanh(mu+np.sqrt(delta0)*gauss_points))**2)
    return gaussian_norm * np.dot (integrand,gauss_weights)

def PrimPhi (mu, delta0):
    integrand = np.log(np.cosh(mu+np.sqrt(delta0)*gauss_points)) * np.tanh(mu+np.sqrt(delta0)*gauss_points)
    return gaussian_norm * np.dot (integrand,gauss_weights)


#### Nested Gaussian intergrals

def InnerPrimPrim (z, mu, delta0, deltainf): # Performs the internal integral
    integrand = np.log(np.cosh(mu+np.sqrt(delta0-deltainf)*gauss_points+np.sqrt(deltainf)*z))
    return gaussian_norm * np.dot (integrand,gauss_weights)

def IntPrimPrim (mu, delta0, deltainf): # Performs the external integral
    integrand = InnerPrimPrim(gauss_points, mu, delta0, deltainf)
    return gaussian_norm * np.dot (integrand**2,gauss_weights)

InnerPrimPrim = np.vectorize(InnerPrimPrim)


def InnerPhiPhi (z, mu, delta0, deltainf):
    integrand = np.tanh(mu+np.sqrt(delta0-deltainf)*gauss_points+np.sqrt(deltainf)*z)
    return gaussian_norm * np.dot (integrand,gauss_weights)

def IntPhiPhi (mu, delta0, deltainf):
    integrand = InnerPhiPhi(gauss_points, mu, delta0, deltainf)
    return gaussian_norm * np.dot (integrand**2,gauss_weights)

InnerPhiPhi = np.vectorize(InnerPhiPhi)


def InnerPrimePrime (z, mu, delta0, deltainf):
    integrand = 1-np.tanh(mu+np.sqrt(delta0-deltainf)*gauss_points+np.sqrt(deltainf)*z)**2
    return gaussian_norm * np.dot (integrand,gauss_weights)

def IntPrimePrime (mu, delta0, deltainf): 
    integrand = InnerPrimePrime(gauss_points, mu, delta0, deltainf)
    return gaussian_norm * np.dot (integrand**2,gauss_weights)

InnerPrimePrime = np.vectorize(InnerPrimePrime)
