import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dsn.util.dsn_util import get_system_from_template, get_arch_from_template
from dsn.train_dsn import train_dsn
import pandas as pd
import scipy.stats
import sys, os

os.chdir("../")

repeats = int(sys.argv[1])
nlayers = int(sys.argv[2])
sigma_init = float(sys.argv[3])
random_seed = int(sys.argv[4])


sysname = "Linear2D"
omega = 1.0
param_dict = {'omega':omega}
system = get_system_from_template(sysname, param_dict)

arch_params = {
               'D':system.D,
               'repeats':repeats,
               'nlayers':nlayers,
               'sigma_init':sigma_init,
              }
param_dict.update(arch_params)
arch_dict = get_arch_from_template(sysname, param_dict)

n = 1000
AL_it_max = 4
c_init_order = -1
AL_fac = 4.0
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
    c_init_order=c_init_order,
    AL_fac=AL_fac,
    min_iters=min_iters,
    max_iters=max_iters,
    random_seed=random_seed,
    lr_order=lr_order,
    check_rate=check_rate,
    dir_str=dir_str,
    db=True,
)
