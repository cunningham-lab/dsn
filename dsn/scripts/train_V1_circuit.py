import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dsn.util.systems import V1Circuit
from dsn.util.dsn_util import get_system_from_template
from dsn.train_dsn import train_dsn
import pandas as pd
import scipy.stats
import sys, os

os.chdir("../")

nlayers = int(sys.argv[1])
c_init_order = int(sys.argv[2])
sigma_init = float(sys.argv[3])
random_seed = int(sys.argv[4])

behavior_type = 'ISN_coeff'
param_dict = {
    "behavior_type":behavior_type,
    "silenced":'V',
}
system = get_system_from_template("V1Circuit", param_dict)

# set up DSN architecture
K = 1
"""
flow_type = 'PlanarFlow';
post_affine = True
arch_dict = {'D':system.D, \
             'K':K, \
             'shared':True, \
             'flow_type':flow_type, \
             'post_affine':post_affine, \
             'repeats':nlayers};
"""
repeats = 1
flow_type = "RealNVP"
real_nvp_arch = {
                 'num_masks':8,
                 'nlayers':nlayers,
                 'upl':20,
                }
mult_and_shift = "post"
arch_dict = {
    "D": system.D,
    "K": K,
    "sigma0":0.1,
    "flow_type": flow_type,
    "real_nvp_arch":real_nvp_arch,
    "repeats": repeats,
    "post_affine": True,
    "shared":False,
}

AL_it_max = 20

batch_size = 300
lr_order = -3
AL_fac = 4.0

iters = 5000

train_dsn(
    system,
    arch_dict,
    batch_size,
    AL_it_max=AL_it_max,
    sigma_init=sigma_init,
    c_init_order=c_init_order,
    AL_fac=AL_fac, 
    min_iters=iters,
    max_iters=iters,
    random_seed=random_seed,
    lr_order=lr_order,
    check_rate=100,
    dir_str="V1Circuit",
    db=False
)
