from train_dsn import train_dsn
import numpy as np
import os
import sys
from tf_util.systems import system_from_str
from matplotlib import pyplot as plt
import scipy.stats

system_str = 'linear_1D'
dt = float(sys.argv[1]);
T = int(sys.argv[2]);

behavior_str = 'steady_state';
random_seed = 0;

TIF_flow_type = 'PlanarFlowLayer';
nlayers = 10;
flow_dict = {'latent_dynamics':None, \
             'TIF_flow_type':TIF_flow_type, \
             'repeats':nlayers};

n = 300;
lr_order = -3;
flow_dict;

init_conds = np.array([1.0]);
system_class = system_from_str(system_str);
system = system_class(behavior_str, dt, T, init_conds;

np.random.seed(0);

mu = np.array([0.1]);
Sigma = np.array([[.1]]);

behavior = {'mu':mu, 'Sigma':Sigma};

X, costs, R2s = train_dsn(system, behavior, T, n, flow_dict, \
	                      lr_order=lr_order, \
	                      random_seed=random_seed);
