import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tf_util.systems import system_from_str
from train_dsn import train_dsn
import sys

planar_layers = int(sys.argv[1])
c_init_order = int(sys.argv[2])
lr_order = int(sys.argv[3])
T = int(sys.argv[4])

c_init = 10 ** c_init_order

system_str = "damped_harmonic_oscillator"
behavior_str = "trajectory"

dt = 0.1

k = 5.0
c = 1.0
m = 1.0
bounds = [0.0, 20.0]
w_0 = np.sqrt(k / m)
zeta = c / (2 * np.sqrt(m * k))
A = 1.0
theta = 0.0
t = np.linspace(0, dt * (T - 1), T)
x = (
    A
    * np.exp(-(zeta * w_0 * t))
    * np.sin(np.sqrt(1 - np.square(zeta)) * w_0 * t + theta)
)
dxdt = A * (
    -zeta
    * w_0
    * np.exp(-zeta * w_0 * t)
    * np.sin(np.sqrt(1.0 - np.square(zeta)) * w_0 * t + theta)
    + np.exp(-zeta * w_0 * t)
    * np.cos(np.sqrt(1 - np.square(zeta)) * w_0 * t + theta)
    * (np.sqrt(1 - np.square(zeta)) * w_0)
)
init_conds = np.array([0.0, dxdt[0]])

system_class = system_from_str(system_str)
system = system_class(behavior_str, T, dt, init_conds, bounds)
print(system.name)

# behavioral constraints
mu = x
Sigma = 0.0001 * np.ones((T,))

behavior = {"mu": mu, "Sigma": Sigma}

random_seed = 0

TIF_flow_type = "PlanarFlowLayer"
nlayers = planar_layers

flow_dict = {
    "latent_dynamics": None,
    "TIF_flow_type": TIF_flow_type,
    "repeats": nlayers,
}

n = 1000
k_max = 20
check_rate = 1000
max_iters = 10000

np.random.seed(0)

cost, phi, T_x = train_dsn(
    system,
    behavior,
    n,
    flow_dict,
    k_max=k_max,
    c_init=c_init,
    lr_order=lr_order,
    check_rate=check_rate,
    max_iters=max_iters,
    random_seed=random_seed,
)
