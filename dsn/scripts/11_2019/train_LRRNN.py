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
sigma_init = float(sys.argv[2])
variance = float(sys.argv[3])
c_init_order = int(sys.argv[4])
random_seed = int(sys.argv[5])

# Real NVP arch params
nlayers = 2
upl = 10

# Get Low Rank RNN system
sysname = "LowRankRNN"
rank = 1
input_type = "input"
behavior_type = "BI"
param_dict = {
    "rank": rank,
    "input_type": input_type,
    "behavior_type": behavior_type,
    "solve_its": 40,
    "variance": variance,
    "gauss_newton": False,
    "solve_eps": 0.2,
}
system = get_system_from_template(sysname, param_dict)

# Get DSN architecture
arch_params = {
    "D": system.D,
    "sigma_init": sigma_init,
    "repeats": repeats,
    "nlayers": nlayers,
    "upl": upl,
}

param_dict.update(arch_params)
arch_dict = get_arch_from_template(system, param_dict)

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
    c_init_order=c_init_order,
    AL_fac=AL_fac,
    min_iters=iters,
    max_iters=iters,
    random_seed=random_seed,
    lr_order=lr_order,
    check_rate=100,
    dir_str="LRRNN",
    db=False,
)
