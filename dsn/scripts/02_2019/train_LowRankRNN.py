import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dsn.util.systems import LowRankRNN
from dsn.train_dsn import train_dsn
import pandas as pd
import scipy.stats
import sys, os

os.chdir("../")

nlayers = int(sys.argv[1])
c_init_order = int(sys.argv[2])
sigma_init = float(sys.argv[3])
random_seed = int(sys.argv[4])


# create an instance of the V1_circuit system class
fixed_params = {}

behavior_type = "struct_chaos"
means = np.array([0.5, 0.5, 0.5])
variances = np.array([0.01, 0.01, 0.01])
behavior = {"type": behavior_type, "means": means, "variances": variances}

# set model options
model_opts = {"rank": 1, "input_type": "spont"}

system = LowRankRNN(
    fixed_params, behavior, model_opts=model_opts, solve_its=25, solve_eps=0.5
)

# set up DSN architecture
flow_type = "PlanarFlow"
post_affine = True
arch_dict = {
    "D": system.D,
    "K": 1, 
    "post_affine": post_affine,
    "flow_type": flow_type,
    "repeats": nlayers,
}

AL_fac = 4.0
AL_it_max = 40
batch_size = 1000
lr_order = -3
min_iters = 1000
max_iters = 2000


train_dsn(
    system,
    arch_dict,
    batch_size,
    AL_it_max=AL_it_max,
    sigma_init=sigma_init,
    c_init_order=c_init_order,
    AL_fac=AL_fac,
    min_iters=min_iters,
    max_iters=max_iters,
    random_seed=random_seed,
    lr_order=lr_order,
    check_rate=100,
    dir_str="test",
)
