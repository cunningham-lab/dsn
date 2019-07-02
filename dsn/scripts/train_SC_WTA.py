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
p = float(sys.argv[2])
inact_str = str(sys.argv[3])
c_init_order = int(sys.argv[4])
sigma_init = float(sys.argv[5])
random_seed = int(sys.argv[6])

AL_it_max = 1

dir_str = 'SC_WTA'
batch_size = 100
AL_fac = 4.0
min_iters=1000
max_iters=1000

nlayers = 10

# create an instance of the V1_circuit system class
fixed_params = {'E_constant':0.0, \
                'E_Pbias':0.1, \
                'E_Prule':0.5, \
                'E_Arule':0.5, \
                'E_choice':-0.2, \
                'E_light':0.1};

C = 1

behavior_type = "WTA"
means = np.array([p, 0.0, 1.0])
if (p==0.0 or p==1.0):
    behavior = {
        "type": behavior_type,
        "means": means,
        "inact_str":inact_str
    }
else:
    behavior = {
        "type": behavior_type,
        "means": means,
        "bounds":np.zeros(C),
        "inact_str":inact_str
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
                      )

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
    db=True,
)


