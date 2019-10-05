import numpy as np
import tensorflow as tf
from dsn.util.systems import V1Circuit
from dsn.util.dsn_util import get_system_from_template, get_arch_from_template
from dsn.train_dsn import train_dsn
import pandas as pd
import scipy.stats
import sys, os

os.chdir("../")

sigma_init = float(sys.argv[1])
repeats = int(sys.argv[2])
nlayers = int(sys.argv[3])
upl = int(sys.argv[4])
c_init_order = int(sys.argv[5])
random_seed = int(sys.argv[6])

silenced = False

# Get V1 system
sysname = "V1Circuit"
behavior_type = 'ISN_coeff'
if silenced:
    param_dict = {
        "behavior_type":behavior_type,
        "silenced":silenced,
    }
else:
    param_dict = {
        "behavior_type":behavior_type,
    }
system = get_system_from_template(sysname, param_dict)

# Get DSN architecture
arch_params = {
               'D':system.D,
               'repeats':repeats,
               'nlayers':nlayers,
               'upl':upl,
               'sigma_init':sigma_init*np.ones((system.D,))
              }

param_dict.update(arch_params)
arch_dict = get_arch_from_template(system, param_dict)

AL_it_max = 20
batch_size = 1000
lr_order = -3
AL_fac = 4.0
iters = 5000

train_dsn(
    system,
    arch_dict,
    batch_size,
    AL_it_max=AL_it_max,
    c_init_order=c_init_order,
    AL_fac=AL_fac, 
    min_iters=iters,
    max_iters=iters,
    random_seed=random_seed,
    lr_order=lr_order,
    check_rate=100,
    dir_str="V1Circuit_test",
    db=False
)
