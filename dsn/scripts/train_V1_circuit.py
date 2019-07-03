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

s = int(sys.argv[1])
nlayers = int(sys.argv[2])
K = int(sys.argv[3])
c_init_order = int(sys.argv[4])
sigma_init = float(sys.argv[5])
random_seed = int(sys.argv[6])
sigma0 = float(sys.argv[7])

behavior_type = 'difference'
fac = 10.0
param_dict = {
    "behavior_type":behavior_type,
    "s":s,
    "fac":fac,
}
system = get_system_from_template("V1Circuit", param_dict)

# set up DSN architecture
flow_type = 'PlanarFlow';
post_affine = True
arch_dict = {'D':system.D, \
             'K':K, \
             'sigma0':sigma0, \
             'flow_type':flow_type, \
             'post_affine':post_affine, \
             'repeats':nlayers};

AL_it_max = 20

batch_size = 200
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
    db=True
)
