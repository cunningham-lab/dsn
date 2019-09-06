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

std = 0.15
var = std**2
AL_it_max = 40

dir_str = 'SC_WTA_%s' % inact_str
batch_size = 300
AL_fac = 4.0
min_iters=5000
max_iters=5000

nlayers = 10
K = 1 
sigma0 = 0.1
N = 500

param_dict = {
    "behavior_type":"WTA",
    "p":p,
    "var":var,
    "inact_str":inact_str,
    "N":N,
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
"""
repeats = 1
flow_type = "RealNVP"
real_nvp_arch = {
                 'num_masks':8,
                 'nlayers':nlayers,
                 'upl':20,
                }
mult_and_shift = "post"
arch_dict = {
    "D": system.D,
    "K": 1,
    "sigma0":0.1,
    "flow_type": flow_type,
    "real_nvp_arch":real_nvp_arch,
    "repeats": repeats,
    "post_affine": True,
    "shared":False,
}
"""

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
    dir_str='SC_WTA_NI',
    savedir=savedir,
    entropy=True,
    db=False,
)


