import numpy as np
import tensorflow as tf
from dsn.util.dsn_util import get_savedir, get_system_from_template, get_arch_from_template
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
upl = int(sys.argv[5])
sigma_init = float(sys.argv[6])
random_seed = int(sys.argv[7])

sysname = "SCCircuit"
dir_str = 'SC_WTA_%s' % inact_str

std = 0.15
var = std**2
N = 500
param_dict = {
    "behavior_type":"WTA",
    "p":p,
    "var":var,
    "inact_str":inact_str,
    "N":N,
    }

system = get_system_from_template(sysname, param_dict)

# Get DSN architecture
repeats = 1
nlayers = 2 
arch_params = {
               'D':system.D,
               'repeats':repeats,
               'nlayers':nlayers,
               'upl':upl,
               'sigma_init':sigma_init,
              }

param_dict.update(arch_params)
arch_dict = get_arch_from_template(system, param_dict)


batch_size = 300
AL_it_max = 40
AL_fac = 4.0
min_iters=5000
max_iters=5000
lr_order = -3

train_dsn(
    system,
    arch_dict,
    batch_size,
    AL_it_max=AL_it_max,
    c_init_order=c_init_order,
    AL_fac=AL_fac,
    min_iters=min_iters,
    max_iters=max_iters,
    random_seed=random_seed,
    lr_order=lr_order,
    check_rate=100,
    dir_str='SC_WTA_NI',
    savedir=None,
    entropy=True,
    db=False,
)


