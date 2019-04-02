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
fixed_params = {'g':0.2, 'betam':0.6, 'betan':1.0}

behavior_type = "CDD"
means = np.array([0.3, 0.3])
variances = np.array([0.001, 0.001])
behavior = {"type": behavior_type, "means": means, "variances": variances}

# set model options
model_opts = {"rank": 2, "input_type": "input"}

solve_its = 100
solve_eps = 0.2
system = LowRankRNN(
    fixed_params, behavior, model_opts=model_opts, solve_its=solve_its, solve_eps=solve_eps
)

# set up DSN architecture
latent_dynamics = None
TIF_flow_type = "PlanarFlow"
mult_and_shift = "post"
arch_dict = {
    "D": system.D,
    "latent_dynamics": latent_dynamics,
    "mult_and_shift": mult_and_shift,
    "TIF_flow_type": TIF_flow_type,
    "repeats": nlayers,
}

k_max = 40
batch_size = 100
lr_order = -3
min_iters = 2500
max_iters = 5000


train_dsn(
    system,
    batch_size,
    arch_dict,
    k_max=k_max,
    sigma_init=sigma_init,
    c_init_order=c_init_order,
    lr_order=lr_order,
    random_seed=random_seed,
    min_iters=min_iters,
    max_iters=max_iters,
    check_rate=200,
    dir_str="test",
)
