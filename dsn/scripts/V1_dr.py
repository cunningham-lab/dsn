import numpy as np
import tensorflow as tf
from epi.models import Parameter, Model
from epi.example_eps import V1_dr_eps
import argparse
import os

# Parse script command-line parameters.
parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=str)
parser.add_argument('--inc_val', type=float)
parser.add_argument('--inc_std', type=float)
parser.add_argument('--num_stages', type=int, default=2)
parser.add_argument('--num_units', type=int, default=25)
parser.add_argument('--logc0', type=float, default=0.) # log10 of c_0
parser.add_argument('--random_seed', type=int, default=1)
args = parser.parse_args()

alpha = args.alpha
inc_val = args.inc_val
inc_std = args.inc_std
num_stages = args.num_stages
num_units = args.num_units
c0 = 10.**args.logc0
random_seed = args.random_seed

# 1. Specify the V1 model for EPI.
D = 4
lb = -5.*np.ones((D,))
ub = 5.*np.ones((D,))
dh = Parameter("dh", D, lb=lb, ub=ub)
parameters = [dh]
model = Model("V1Circuit", parameters)

# 2. Define the emergent property.
# Emergent property statistics (eps).
dr = V1_dr_eps(alpha, inc_val)
model.set_eps(dr)

# Emergent property values.
mu = np.array([inc_val, inc_std**2])

# 3. Run EPI.
init_params = {'loc':0., 'scale':2.}
q_theta, opt_data, save_path, failed = model.epi(
    mu, 
    arch_type='coupling', 
    num_stages=num_stages, 
    num_layers=2,
    num_units=num_units,
    post_affine=False,
    batch_norm=False,
    init_params=init_params,
    K=2,
    N=500, 
    num_iters=500, 
    lr=1e-3, 
    c0=1e0,
    beta=4.,
    nu=0.2,
    random_seed=random_seed,
    verbose=True,
    stop_early=True,
    log_rate=100,
    save_movie_data=True,
)

if not failed:
    model.epi_opt_movie(save_path)
