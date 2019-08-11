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
sigma0 = float(sys.argv[5])
random_seed = int(sys.argv[6])

D = 4
"""
flow_type = "PlanarFlow"
mult_and_shift = "post"
arch_dict = {
    "D": D,
    "K": K,
    "sigma0":sigma0,
    "flow_type": flow_type,
    "repeats": nlayers,
    "post_affine": True,
    "shared":False,
}
"""

repeats = 1
flow_type = "RealNVP"
real_nvp_arch = {
                 'num_masks':4,
                 'nlayers':nlayers,
                 'upl':10,
                }
mult_and_shift = "post"
arch_dict = {
    "D": D,
    "K": K,
    "sigma0":sigma0,
    "flow_type": flow_type,
    "real_nvp_arch":real_nvp_arch,
    "repeats": repeats,
    "post_affine": True,
    "shared":False,
}

fixed_params = {"tau": 1.0}

omega = 1
means = np.array([0.0, 2 * np.pi * omega])
Sigma = np.array([1.0, 1.0])
behavior = {"type": "oscillation", "means": means, "variances": Sigma}

system = Linear2D(fixed_params, behavior)


n = 1000
AL_it_max = 5 
lr_order = -3
min_iters = 2000
max_iters = 2000
check_rate = 100
dist_seed = 0
dir_str = "LDS_test"


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
