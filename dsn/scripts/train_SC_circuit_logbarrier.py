import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dsn.util.systems import SCCircuit
from dsn.train_dsn import train_dsn
import pandas as pd
import scipy.stats
import sys, os

os.chdir("../")

p = float(sys.argv[1])
param_str = str(sys.argv[2])
c_init_order = int(sys.argv[3])
random_seed = int(sys.argv[4])

nlayers = 5
sigma_init = 1.0

# create an instance of the V1_circuit system class
fixed_params = {'E_constant':0.0, \
                'E_Pbias':0.1, \
                'E_Prule':0.5, \
                'E_Arule':0.5, \
                'E_choice':-0.2, \
                'E_light':0.1};

behavior_type = "feasible"

means = np.array([0.1])
variances = np.array([0.0002])
behavior = {
    "type": behavior_type,
    "means": means,
    "variances":variances,
}

model_opts = {"params":param_str}
system = SCCircuit(fixed_params, behavior, model_opts)

# set up DSN architecture
latent_dynamics = None;
TIF_flow_type = 'PlanarFlow';
mult_and_shift = 'post';
arch_dict = {'D':system.D, \
             'latent_dynamics':latent_dynamics, \
             'mult_and_shift':mult_and_shift, \
             'TIF_flow_type':TIF_flow_type, \
             'repeats':nlayers};


k_max = 4

batch_size = 100
lr_order = -3


train_dsn(
    system,
    batch_size,
    arch_dict,
    k_max=k_max,
    sigma_init=sigma_init,
    c_init_order=c_init_order,
    lr_order=lr_order,
    random_seed=random_seed,
    min_iters=500,
    max_iters=2000,
    check_rate=100,
    dir_str='logbarrier',
)