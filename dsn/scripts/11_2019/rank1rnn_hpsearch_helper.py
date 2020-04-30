import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tf_util.systems import system_from_str
from train_dsn import train_dsn
import sys
from util import fct_integrals as integrals
from util import tf_integrals as tf_integrals
from util import fct_mf as mf


def compute_bistable_mu(Sini, ics_0, ics_1):
    ### Set parameters

    Mm = 3.5  # Mean of m
    Mn = 1.0  # Mean of n
    Mi = 0.0  # Mean of I

    Sim = 1.0  # Std of m
    Sin = 1.0  # Std of n
    Sip = 1.0  # Std of input orthogonal to m and n, along h (see Methods)

    g = 0.8
    tol = 1e-10

    eps = 0.2

    ParVec = [Mm, Mn, Mi, Sim, Sin, Sini, Sip]
    ys0, count = mf.SolveStatic(ics_0, g, ParVec, eps, tol)
    ys1, count = mf.SolveStatic(ics_1, g, ParVec, eps, tol)

    ss0 = ys0[-1, 2]
    ss1 = ys1[-1, 2]
    mu = np.array([ss0, ss1])
    return mu


planar_layers = int(sys.argv[1])
num_neurons = int(sys.argv[2])
c_init_order = int(sys.argv[3])
lr_order = int(sys.argv[4])
T = int(sys.argv[5])

D = 2 * num_neurons
c_init = 10 ** c_init_order
n = 1000
K = 1
M = n
k_max = 10
check_rate = 1000
max_iters = 5000

system_str = "rank1_rnn"
behavior_str = "bistable"

Sini = 0.5

ics_0 = np.array([5.0, 5.0, 5.0], np.float64)
ics_1 = np.array([-5.0, 5.0, -5.0], np.float64)

Ics_0 = np.tile(np.expand_dims(np.expand_dims(ics_0, 0), 1), [K, M, 1])
Ics_1 = np.tile(np.expand_dims(np.expand_dims(ics_1, 0), 1), [K, M, 1])

system_class = system_from_str(system_str)
system = system_class(D, T, Sini, Ics_0, Ics_1, behavior_str)

# behavioral constraints
mu = compute_bistable_mu(Sini, ics_0, ics_1)
Sigma = 0.01 * np.ones((2,))
behavior = {"mu": mu, "Sigma": Sigma}

random_seed = 0

TIF_flow_type = "PlanarFlowLayer"
nlayers = planar_layers

flow_dict = {
    "latent_dynamics": None,
    "TIF_flow_type": TIF_flow_type,
    "repeats": nlayers,
}

np.random.seed(0)

cost, phi, T_x = train_dsn(
    system,
    behavior,
    n,
    flow_dict,
    k_max=k_max,
    c_init=c_init,
    lr_order=lr_order,
    check_rate=check_rate,
    max_iters=max_iters,
    random_seed=random_seed,
)
