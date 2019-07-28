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

isn_str = str(sys.argv[1])
silenced = str(sys.argv[2])
c_init_order = int(sys.argv[3])
sigma_init = float(sys.argv[4])
random_seed = int(sys.argv[5])

sysname = 'V1Circuit'
behavior_type = "difference"
param_dict = {'behavior_type':behavior_type,
              'ISN':isn_str,
              'silenced':silenced}
system = get_system_from_template(sysname, param_dict)

# set up DSN architecture
K = 1
flow_type = 'PlanarFlow';
nlayers = 10
post_affine = True
arch_dict = {'D':system.D, \
             'K':K, \
             'shared':True, \
             'flow_type':flow_type, \
             'post_affine':post_affine, \
             'repeats':nlayers};

AL_it_max = 20

batch_size = 400
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
