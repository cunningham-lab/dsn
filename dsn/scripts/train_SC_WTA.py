import numpy as np
import tensorflow as tf
from dsn.util.dsn_util import get_savedir, get_system_from_template
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

std = 0.05
var = std**2
AL_it_max = 20

dir_str = 'test_SC_WTA_%s' % inact_str
batch_size = 300
AL_fac = 4.0
min_iters=5000
max_iters=5000

nlayers = 10
K = 20
sigma0 = 0.1

param_dict = {
    "behavior_type":"WTA",
    "p":p,
    "var":var,
    "inact_str":inact_str,
    }

system = get_system_from_template('SCCircuit', param_dict)

# set up DSN architecture
flow_type = 'PlanarFlow';
arch_dict = {'D':system.D, \
             'K':K, \
             'sigma0':sigma0, \
             'post_affine':True, \
             'shared':True, \
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
    db=False,
)


