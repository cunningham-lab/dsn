import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dsn.util.systems import STGCircuit
from dsn.train_dsn import train_dsn
import pandas as pd
import scipy.stats
import sys, os

os.chdir("../")

freq = str(sys.argv[1])
nlayers = int(sys.argv[2])
c_init_order = int(sys.argv[3])
sigma_init = float(sys.argv[4])
random_seed = int(sys.argv[5])

behavior_type = "hubfreq"

if (freq == "high"):
    T = 500 
    mean = 0.725
    variance = (.025)**2
elif (freq == "med"):
    T = 200
    mean = 0.525
    variance = (.025)**2
else:
    print('Error: freq not med or high.')
    exit()

dt = 0.025
fft_start = 0
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
    arch_dict,
    batch_size,
    k_max=k_max,
    sigma_init=sigma_init,
    c_init_order=c_init_order,
    AL_fac=4.0,
    min_iters=1000,
    max_iters=2000,
    random_seed=random_seed,
    lr_order=lr_order,
    check_rate=100,
    dir_str="STGCircuit",
    db=False,
)
