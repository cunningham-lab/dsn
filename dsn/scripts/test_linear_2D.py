import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dsn.util.systems import system_from_str
from dsn.train_dsn import train_dsn
import seaborn as sns
import pandas as pd
import scipy.stats
import sys, os

os.chdir("../")

nlayers = int(sys.argv[1]);
c_init_order = int(sys.argv[2]);
sigma_init = float(sys.argv[3]);
random_seed = int(sys.argv[4]);

system_str = 'linear_2D';
D = 4;
T = 1;

latent_dynamics = None;
TIF_flow_type = 'PlanarFlowLayer';
scale_layer = True;
flow_dict = {'latent_dynamics':latent_dynamics, \
             'TIF_flow_type':TIF_flow_type, \
             'repeats':nlayers, \
             'scale_layer':scale_layer};

n = 1000;
k_max = 10;
lr_order = -3;
min_iters = 5000;
max_iters = 10000;
check_rate = 100;
dist_seed = 0;
dir_str = 'convergence_testing';

behavior_str = 'oscillation';
K = 1;
system_class = system_from_str(system_str);
system = system_class(behavior_str);

np.random.seed(dist_seed);

omega = 2;
mu = np.array([0.0, 2*np.pi*omega]);
Sigma = np.array([1.0, 1.0]);
behavior = {'mu':mu, 'Sigma':Sigma};

print('Behavior:');
print('mu');
print(mu);
print('Sigma');
print(Sigma[0]);
print(Sigma[1]);

cost, phi, T_x = train_dsn(system, behavior, n, flow_dict, \
                       k_max=k_max, sigma_init=sigma_init, c_init_order=c_init_order, lr_order=lr_order,\
                       random_seed=random_seed, min_iters=min_iters, max_iters=max_iters, \
                       check_rate=check_rate, dir_str=dir_str);
