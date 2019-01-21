import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dsn.util.systems import V1Circuit
from dsn.train_dsn import train_dsn
import pandas as pd
import scipy.stats
import sys, os

os.chdir("../")

nlayers = int(sys.argv[1]);
c_init_order = int(sys.argv[2]);
sigma_init = float(sys.argv[3]);
random_seed = int(sys.argv[4]);

D = 8
latent_dynamics = None;
TIF_flow_type = 'PlanarFlow';
elem_mult_flow = True;
arch_dict = {'D':D, \
             'latent_dynamics':latent_dynamics, \
             'elem_mult_flow':elem_mult_flow, \
             'TIF_flow_type':TIF_flow_type, \
             'repeats':nlayers};

# create an instance of the V1_circuit system class
fixed_params = {'W_EE':1.0, \
                'W_PE':1.0, \
                'W_SE':1.0, \
                'W_VE':1.0, \
                'h_FFE':0.0, \
                'h_FFP':0.0, \
                'h_LATE':0.0, \
                'h_LATP':0.0, \
                'h_LATS':0.0, \
                'h_LATV':0.0, \
                'tau':1.0, \
                'n':2.0, \
                's_0':30};

behavior_type = 'difference'
diff_inds = [2, 3]
c_vals=np.array([0.0])
s_vals=np.array([1.0])
r_vals=np.array([0.0, 1.0])
d_mean = np.array([1.0, 0.0]);
d_vars = np.array([0.1, 0.1]);
behavior = {'type':behavior_type, \
            'diff_inds':diff_inds, \
            'c_vals':c_vals, \
            's_vals':s_vals, \
            'r_vals':r_vals, \
            'd_mean':d_mean, \
            'd_var':d_vars}

# set model options
model_opts = {'g_FF':'c', 'g_LAT':'linear', 'g_RUN':'r'}

T = 40
dt = 0.25
init_conds = np.expand_dims(np.array([1.0, 1.1, 1.2, 1.3]), 1)

system = V1Circuit(fixed_params, behavior, model_opts, T, dt, init_conds)

k_max = 10
batch_size = 1000;
lr_order = -3


train_dsn(system, batch_size, arch_dict, \
          k_max=k_max, sigma_init=sigma_init, c_init_order=c_init_order, lr_order=lr_order,\
          random_seed=random_seed, min_iters=1000, max_iters=2000, \
          check_rate=100, dir_str='test')
