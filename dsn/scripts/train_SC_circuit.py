import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dsn.util.systems import SCCircuit
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
fixed_params = {'E_constant':0.0, \
                'E_Pbias':0.1, \
                'E_Prule':0.5, \
                'E_Arule':0.5, \
                'E_choice':-0.2, \
                'E_light':0.1};


p = 0.8
pvar = 0.01
means = np.array([p, 0.0])
variances = np.array([pvar, 0.0])
behavior = {
    "type": "standard",
    "means": means,
    "variances": variances,
}
model_opts = {"params":"full"}
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


k_max = 40

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
    min_iters=1000,
    max_iters=2000,
    check_rate=100,
    dir_str="SCCircuit",
)
