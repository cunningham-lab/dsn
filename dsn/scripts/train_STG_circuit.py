import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dsn.util.dsn_util import get_system_from_template, get_arch_from_template
from dsn.train_dsn import train_dsn
import pandas as pd
import scipy.stats
import sys, os

os.chdir("../")

freq = str(sys.argv[1])
num_masks = int(sys.argv[2])
nlayers = int(sys.argv[3])
c_init_order = int(sys.argv[4])
sigma_init = float(sys.argv[5])
random_seed = int(sys.argv[6])

# Get STG system
sysname = "STGCircuit"
param_dict = {"freq":freq}
system = get_system_from_template(sysname, param_dict)

repeats = 1
# Get DSN architecture
arch_params = {
               'D':system.D,
               'repeats':repeats,
               'num_masks':num_masks,
               'nlayers':nlayers,
               'sigma_init':sigma_init,
              }
param_dict.update(arch_params)
arch_dict = get_arch_from_template(system, param_dict)

AL_it_max = 10
AL_fac = 4.0
iters = 2000
batch_size = 300 
lr_order = -3
check_rate = 100


train_dsn(
    system,
    arch_dict,
    n=batch_size,
    AL_it_max=AL_it_max,
    c_init_order=c_init_order,
    AL_fac=AL_fac,
    min_iters=iters,
    max_iters=iters,
    random_seed=random_seed,
    lr_order=lr_order,
    check_rate=check_rate,
    dir_str="STGCircuit_%.2f" % sigma_init,
    db=False,
)
