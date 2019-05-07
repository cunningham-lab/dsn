import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dsn.util.systems import STGCircuit
from dsn.train_dsn import train_dsn
import pandas as pd
import scipy.stats
import sys, os

os.chdir("../")

nlayers = int(sys.argv[1])
c_init_order = int(sys.argv[2])
sigma_init = float(sys.argv[3])
random_seed = int(sys.argv[4])

behavior_type = "hubfreq"

dt = 0.025
T = 210
fft_start = 10
w = 20

mean = 0.55
variance = 0.0001
fixed_params = {'g_synB':5e-9}
behavior = {"type":"hubfreq",
            "mean":mean,
            "variance":variance}
model_opts = {"dt":dt,
              "T":T,
              "fft_start":fft_start,
              "w":w
             }

system = STGCircuit(fixed_params, behavior, model_opts)

# set up DSN architecture
latent_dynamics = None;
TIF_flow_type = 'PlanarFlow';
mult_and_shift = 'post';
arch_dict = {'D':system.D, \
             'latent_dynamics':latent_dynamics, \
             'mult_and_shift':mult_and_shift, \
             'TIF_flow_type':TIF_flow_type, \
             'repeats':nlayers};


k_max = 20

batch_size = 1000
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
    dir_str="STGCircuit",
    db=True,
)
