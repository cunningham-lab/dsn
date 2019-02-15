import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dsn.util.systems import system_from_str
from dsn.train_dsn import train_dsn
import seaborn as sns
import pandas as pd
import scipy.stats
import sys

D = int(sys.argv[1])
c_init_order = int(sys.argv[2])
dist_seed = int(sys.argv[3])
random_seed = int(sys.argv[4])

system_str = "normal"
T = 1

TIF_flow_type = "PlanarFlowLayer"
nlayers = 4
flow_dict = {
    "latent_dynamics": None,
    "TIF_flow_type": TIF_flow_type,
    "repeats": nlayers,
    "scale_layer": True,
}

n = 1000
k_max = 10
sigma_init = 10
lr_order = -3
min_iters = 50000
max_iters = 100000
check_rate = 100
dir_str = "convergence_testing"

K = 1
system_class = system_from_str(system_str)
system = system_class(D, T)

np.random.seed(dist_seed)
mu = np.zeros((D,))
df_fac = 15
df = df_fac * D
Sigma_dist = scipy.stats.invwishart(df=df, scale=df * np.eye(D))
Sigma = Sigma_dist.rvs(1)

behavior = {"mu": mu, "Sigma": Sigma}

print("Behavior:")
print("mu")
print(mu)
print("Sigma")
print(Sigma[0])
print(Sigma[1])

cost, phi, T_x = train_dsn(
    system,
    behavior,
    n,
    flow_dict,
    k_max=k_max,
    sigma_init=sigma_init,
    c_init_order=c_init_order,
    lr_order=lr_order,
    random_seed=random_seed,
    min_iters=min_iters,
    max_iters=max_iters,
    check_rate=check_rate,
    dir_str=dir_str,
)
