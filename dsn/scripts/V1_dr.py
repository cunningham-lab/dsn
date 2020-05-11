import numpy as np
import tensorflow as tf
from epi.models import Parameter, Model
from epi.example_eps import V1_dr_eps
import argparse
import os

# Parse script command-line parameters.
parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=str)
parser.add_argument('--beta', type=str, default='')
parser.add_argument('--inc_val', type=float)
parser.add_argument('--inc_std', type=float)
parser.add_argument('--num_stages', type=int, default=2)
parser.add_argument('--num_units', type=int, default=50)
parser.add_argument('--logc0', type=float, default=0.) # log10 of c_0
parser.add_argument('--random_seed', type=int, default=1)
args = parser.parse_args()

alpha = args.alpha
beta = args.beta
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

beta_str = ''
if beta == '':
    b = np.array([1., 1., 1., 1.25])
elif beta == 'P':
    b = np.array([1., -5., 1., 1.25])
    beta_str += '_P'
elif beta == 'S':
    b = np.array([1., 1., -5., 1.25])
    beta_str += '_S'
else:
    raise(NotImplentedError("Error: beta = %s ?" % beta))
    

name = "V1Circuit_%s%s" % (alpha, beta_str)
model = Model(name, parameters)

# 2. Define the emergent property.
# Emergent property statistics (eps).
dr = V1_dr_eps(alpha, inc_val, b=b)
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
    post_affine=True,
    batch_norm=True,
    init_params=init_params,
    K=15,
    N=500, 
    num_iters=5000, 
    lr=1e-3, 
    c0=c0,
    beta=4.,
    nu=1.0,
    random_seed=random_seed,
    verbose=True,
    stop_early=True,
    log_rate=250,
    save_movie_data=True,
)

if not failed:
    print("Making movie.")
    model.epi_opt_movie(save_path)
    print("done.")

