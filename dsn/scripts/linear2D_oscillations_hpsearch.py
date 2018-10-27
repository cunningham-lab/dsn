import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tf_util.systems import system_from_str
from train_dsn import train_dsn

system_D = 2;
system_str = 'linear_%dD' % system_D;

behavior_str = 'oscillation';

system_class = system_from_str(system_str);
system = system_class(behavior_str);
print(system.name)

# behavioral constraints
mu = np.array([0.0, 0.0, 16.0]);
Sigma = np.array([.001, .001, 0.1]);

behavior = {'mu':mu, 'Sigma':Sigma};

random_seed = 0;

TIF_flow_type = 'PlanarFlowLayer';
nlayers = 4;
flow_dict = {'latent_dynamics':None, \
             'TIF_flow_type':TIF_flow_type, \
             'repeats':nlayers};

n = 1000;
k_max = 10;
c_init = 1e4;
lr_order = -3;
check_rate = 100;
max_iters = 1000;

np.random.seed(0);

cost, phi, T_x = train_dsn(system, behavior, n, flow_dict, \
                       k_max=k_max, c_init=c_init, lr_order=lr_order, check_rate=check_rate, \
                       max_iters=max_iters, random_seed=random_seed);

