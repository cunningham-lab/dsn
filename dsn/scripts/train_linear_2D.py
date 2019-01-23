import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dsn.util.systems import Linear2D
from dsn.train_dsn import train_dsn
import pandas as pd
import scipy.stats
import sys, os

os.chdir('../')

nlayers = int(sys.argv[1]);
c_init_order = int(sys.argv[2]);
sigma_init = float(sys.argv[3]);
random_seed = int(sys.argv[4]);

D = 4;
T = 1;

latent_dynamics = None;
TIF_flow_type = 'PlanarFlow';
mult_and_shift = 'post';
arch_dict = {'D':D, \
             'latent_dynamics':latent_dynamics, \
             'mult_and_shift':mult_and_shift, \
             'TIF_flow_type':TIF_flow_type, \
             'repeats':nlayers};

n = 1000;
k_max = 20;
lr_order = -3;
min_iters = 10000;
max_iters = 20000;
check_rate = 100;
dist_seed = 0;
dir_str = 'Linear2D/1Hz/d';


fixed_params = {'tau':1.0}

omega = 1
mu = np.array([0.0, 2*np.pi*omega]);
Sigma = np.array([1.0, 1.0]);
behavior = {'type':'oscillation', 'means':mu, 'variances':Sigma};

system = Linear2D(fixed_params, behavior);

np.random.seed(dist_seed);
cost, z = train_dsn(system, n, arch_dict, \
                       k_max=k_max, sigma_init=sigma_init, c_init_order=c_init_order, lr_order=lr_order,\
                       random_seed=random_seed, min_iters=min_iters, max_iters=max_iters, \
                       check_rate=check_rate, dir_str=dir_str);
