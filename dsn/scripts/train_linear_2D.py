import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dsn.util.systems import Linear2D
from dsn.train_dsn import train_dsn
import pandas as pd
import scipy.stats
import sys, os

os.chdir("../")

nlayers = int(sys.argv[1])
c_init_order = int(sys.argv[2])
K = int(sys.argv[3])
sigma_init = float(sys.argv[4])
random_seed = int(sys.argv[5])

D = 4

flow_type = "PlanarFlow"
mult_and_shift = "post"
arch_dict = {
    "D": D,
    "K": K,
    "flow_type": flow_type,
    "repeats": nlayers,
    "post_affine": True,
}



fixed_params = {"tau": 1.0}

omega = 1
mu = np.array([0.0, 2 * np.pi * omega])
Sigma = np.array([1.0, 1.0])
behavior = {"type": "oscillation", "means": mu, "variances": Sigma}

system = Linear2D(fixed_params, behavior)


n = 1000
AL_it_max = 2
lr_order = -3
min_iters = 1000
max_iters = 1000
check_rate = 10
dist_seed = 0
dir_str = "Linear2D/"


np.random.seed(dist_seed)
cost, z = train_dsn(
    system,
    arch_dict,
    n=n,
    AL_it_max=AL_it_max,
    sigma_init=sigma_init,
    c_init_order=c_init_order,
    AL_fac=4.0,
    min_iters=min_iters,
    max_iters=max_iters,
    random_seed=random_seed,
    lr_order=lr_order,
    check_rate=check_rate,
    dir_str=dir_str,
    db=True,
)
