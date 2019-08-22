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
K = int(sys.argv[4])
sigma_init = float(sys.argv[5])
random_seed = int(sys.argv[6])

if (K > 1):
    sigma0 = float(sys.argv[7])

behavior_type = "freq"

if (freq == "med"):
    T = 500
    mean = 0.525*np.ones((5,))
    variance = (.025)**2*np.ones((5,))
else:
    print('Error: freq not med or high.')
    exit()

dt = 0.025
fft_start = 0
w = 20

fixed_params = {'g_synB':5e-9}
behavior = {"type":behavior_type,
            "mean":mean,
            "variance":variance}
model_opts = {"dt":dt,
              "T":T,
              "fft_start":fft_start,
              "w":w
             }

system = STGCircuit(fixed_params, behavior, model_opts)

repeats = 1
flow_type = "RealNVP"
real_nvp_arch = {
                 'num_masks':2,
                 'nlayers':nlayers,
                 'upl':10,
                }
arch_dict = {
    "D": system.D,
    "K": K,
    "sigma0":sigma0,
    "flow_type": flow_type,
    "real_nvp_arch":real_nvp_arch,
    "repeats": repeats,
    "post_affine": True,
    "shared":False,
}


k_max = 10

batch_size = 200
lr_order = -3

iters = 10000

train_dsn(
    system,
    arch_dict,
    n=batch_size,
    AL_it_max=k_max,
    sigma_init=sigma_init,
    c_init_order=c_init_order,
    AL_fac=4.0,
    min_iters=iters,
    max_iters=iters,
    random_seed=random_seed,
    lr_order=lr_order,
    check_rate=100,
    dir_str="STGCircuit_big",
    db=True,
)
