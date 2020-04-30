import numpy as np
import tensorflow as tf
from dsn.util.dsn_util import get_savedir
from dsn.util.systems import SCCircuit
from dsn.train_dsn import train_dsn
import pandas as pd
import scipy.stats
import sys, os

os.chdir("../")

param_str = str(sys.argv[1])
randsearch = int(sys.argv[2])
random_seed = int(sys.argv[3])
if (len(sys.argv) > 4):
    c_init_order = int(sys.argv[4])
    sigma_init = float(sys.argv[5])

AL_it_max = 40

dir_str = 'SC_test'
if (randsearch == 1): 
    np.random.seed(random_seed)
    dir_str = dir_str + '_randsearch'
    sigma_init =  np.around(np.random.uniform(1.0, 6.0), 2)
    batch_size = np.random.randint(200, 500)
    c_init_order =  np.random.uniform(1.0, 10.0)
    AL_fac = np.random.uniform(2.0, 10.0)
    max_iters = np.random.randint(5000, 20000)
    min_iters = max_iters
elif (randsearch == 0):
    batch_size = 200
    AL_fac = 4.0
    min_iters=5000
    max_iters=5000
else:
    print('Error: randsearch must be 0 or 1.')
    exit()

nlayers = 10

# create an instance of the V1_circuit system class
fixed_params = {'E_constant':0.0, \
                'E_Pbias':0.1, \
                'E_Prule':0.5, \
                'E_Arule':0.5, \
                'E_choice':-0.2, \
                'E_light':0.1};
#fixed_params = {}

C = 2

if (C==2):
    p_NI = 0.8
    p_DI = 0.6
    behavior_type = "inforoute"
    means = np.array([p_NI, p_DI, 0.0, 0.0, 1.0, 1.0])
elif (C==4):
    err_inc_P = 0.05
    err_inc_A = 0.2
    behavior_type = "inforoute"
    means = np.array([err_inc_P, err_inc_A, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
elif (C==6):
    err_inc_P_DI = 0.05
    err_inc_A_DI = 0.2
    err_inc_P_CI = 0.0
    err_inc_A_CI = 0.0
    behavior_type = "inforoute"
    means = np.array([err_inc_P_DI, err_inc_P_CI, err_inc_A_DI, err_inc_A_CI, \
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

behavior = {
    "type": behavior_type,
    "means": means,
    "bounds":np.zeros(C),
}

model_opts = {"params":param_str, "C":C}
system = SCCircuit(fixed_params, behavior, model_opts)

# set up DSN architecture
latent_dynamics = None;
flow_type = 'PlanarFlow';
arch_dict = {'D':system.D, \
             'K':1,
             'post_affine':True, \
             'flow_type':flow_type, \
             'repeats':nlayers};

lr_order = -3

savedir = get_savedir(system, 
                      arch_dict, 
                      sigma_init, 
                      c_init_order, 
                      random_seed, 
                      dir_str, 
                      randsearch=randsearch
                      )


print('-- Hyperparameters --')
print('batch_size:', batch_size)
print('sigma_init:', sigma_init)
print('c_init_order:', c_init_order)
print('AL_fac:', AL_fac)
print('epoch iters:', max_iters)

train_dsn(
    system,
    arch_dict,
    batch_size,
    AL_it_max=AL_it_max,
    sigma_init=sigma_init,
    c_init_order=c_init_order,
    AL_fac=AL_fac,
    min_iters=min_iters,
    max_iters=max_iters,
    random_seed=random_seed,
    lr_order=lr_order,
    check_rate=100,
    dir_str='test',
    savedir=savedir,
    entropy=True,
    db=False,
)


