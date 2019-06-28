import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dsn.util.systems import V1Circuit
from dsn.train_dsn import train_dsn
import pandas as pd
import scipy.stats
import sys, os

os.chdir("../")

s = int(sys.argv[1])
nlayers = int(sys.argv[2])
K = int(sys.argv[3])
c_init_order = int(sys.argv[4])
sigma_init = float(sys.argv[5])
random_seed = int(sys.argv[6])
sigma0 = float(sys.argv[7])

# create an instance of the V1_circuit system class
base_I = 0.15
run_I = 0.3
tau = 0.15

fixed_params = {'b_E':base_I, \
                'b_P':base_I, \
                'b_S':base_I, \
                'b_V':base_I, \
                'h_RUNE':run_I, \
                'h_RUNP':run_I, \
                'h_RUNS':run_I, \
                'h_RUNV':run_I, \
                'h_FFE':0.0, \
                'h_FFP':0.0, \
                'h_LATE':0.0, \
                'h_LATP':0.0, \
                'h_LATS':0.0, \
                'h_LATV':0.0, \
                'tau':tau, \
                'n':2.0, \
                's_0':30};

behavior_type = "difference"

c_vals=np.array([1.0])
s_vals=np.array([s])
r_vals=np.array([0.0, 1.0])
fac = 10.0
C = c_vals.shape[0]*s_vals.shape[0]*r_vals.shape[0]
bounds = np.zeros((C*4,))
behavior = {'type':behavior_type, \
            'c_vals':c_vals, \
            's_vals':s_vals, \
            'r_vals':r_vals, \
            'fac':fac,
            'bounds':bounds}

# set model options
model_opts = {"g_FF": "c", "g_LAT": "square", "g_RUN": "r"} 
T = 50
dt = 0.05
init_conds = np.expand_dims(np.array([1.0, 1.1, 1.2, 1.3]), 1)

system = V1Circuit(fixed_params, behavior, model_opts, T, dt, init_conds)

# set up DSN architecture
flow_type = 'PlanarFlow';
post_affine = True
arch_dict = {'D':system.D, \
             'K':K, \
             'sigma0':sigma0, \
             'flow_type':flow_type, \
             'post_affine':post_affine, \
             'repeats':nlayers};

AL_it_max = 1

batch_size = 200
lr_order = -3
AL_fac = 4.0

iters = 200

train_dsn(
    system,
    arch_dict,
    batch_size,
    AL_it_max=AL_it_max,
    sigma_init=sigma_init,
    c_init_order=c_init_order,
    AL_fac=AL_fac, 
    min_iters=iters,
    max_iters=iters,
    random_seed=random_seed,
    lr_order=lr_order,
    check_rate=10,
    dir_str="V1Circuit",
    db=True
)
