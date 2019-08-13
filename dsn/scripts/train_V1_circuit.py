import numpy as np
import tensorflow as tf
from dsn.util.systems import V1Circuit
from dsn.util.dsn_util import get_system_from_template
from dsn.train_dsn import train_dsn
import pandas as pd
import scipy.stats
import sys, os

os.chdir("../")

silenced = str(sys.argv[1])
nlayers = int(sys.argv[2])
c_init_order = int(sys.argv[3])
random_seed = int(sys.argv[4])

behavior_type = 'ISN_coeff'
param_dict = {
    "behavior_type":behavior_type,
    "silenced":silenced,
}
system = get_system_from_template("V1Circuit", param_dict)

# set up DSN architecture
K = 1
repeats = 2
flow_type = "RealNVP"
real_nvp_arch = {
                 'num_masks':8,
                 'nlayers':nlayers,
                 'upl':10,
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
