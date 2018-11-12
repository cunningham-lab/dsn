import numpy as np
import tensorflow as tf
from dsn.util.systems import system_from_str
from dsn.train_dsn import train_dsn
from dsn.util import fct_integrals as integrals
from dsn.util import tf_integrals as tf_integrals
from dsn.util import fct_mf as mf
import sys
import os

os.chdir("../")

nlayers = int(sys.argv[1]);
sigma_init = float(sys.argv[2]);
c_init_order = int(sys.argv[3]);
random_seed = int(sys.argv[4]);

TIF_flow_type = 'PlanarFlowLayer';
flow_dict = {'latent_dynamics':None, \
             'TIF_flow_type':TIF_flow_type, \
             'repeats':nlayers, \
             'scale_layer':True};

n = 1000;
k_max = 20;
check_rate = 100;
min_iters = 1000;
max_iters = 2000;
lr_order = -3;

def compute_bistable_mu(Sini, ics_0, ics_1):
    ### Set parameters

    Mm = 3.5      # Mean of m
    Mn = 1.       # Mean of n
    Mi = 0.       # Mean of I

    Sim = 1.      # Std of m
    Sin = 1.      # Std of n
    Sip = 1.      # Std of input orthogonal to m and n, along h (see Methods)

    g = 0.8
    tol = 1e-10;
    
    eps = 0.2;
    
    ParVec = [Mm, Mn, Mi, Sim, Sin, Sini, Sip];
    ys0, count = mf.SolveStatic(ics_0, g, ParVec, eps, tol);
    ys1, count = mf.SolveStatic(ics_1, g, ParVec, eps, tol);
    
    ss0 = ys0[-1,2];
    ss1 = ys1[-1,2];
    mu = np.array([ss0, ss1]);
    return mu;

system_str = 'rank1_rnn_std';

K = 1;
M = n;
behavior_str = 'bistable';
T = 10;
ics_0 = np.array([5., 5., 5.], np.float64);
ics_1 = np.array([-5., 5., -5.], np.float64);

Ics_0 = np.tile(np.expand_dims(np.expand_dims(ics_0, 0), 1), [K,M,1]);
Ics_1 = np.tile(np.expand_dims(np.expand_dims(ics_1, 0), 1), [K,M,1]);

system_class = system_from_str(system_str);
system = system_class(T, Ics_0, Ics_1, behavior_str);

Sini1 = 0.5;
mu1 = compute_bistable_mu(Sini1, ics_0, ics_1);
print('mu1', mu1);
Sini2 = 1.0;
mu2 = compute_bistable_mu(Sini2, ics_0, ics_1);
print('mu2', mu2);

mu = np.concatenate((mu1, mu2), axis=0);

Sigma = 0.05*np.ones((mu.shape[0],));
behavior = {'mu':mu, 'Sigma':Sigma};
print(behavior);
cost, phi, T_x = train_dsn(system, behavior, n, flow_dict, \
                       k_max=k_max, sigma_init=sigma_init, \
                       c_init_order=c_init_order, lr_order=lr_order, \
                       random_seed=random_seed, min_iters=min_iters, \
                       max_iters=max_iters, check_rate=check_rate);

