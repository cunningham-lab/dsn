import matplotlib.pyplot as plt
import numpy as np
import sys, os
from dsn.util.dsn_util import get_savedir
from dsn.util.systems import Linear2D
from dsn.util.plot_util import make_training_movie
import time

os.chdir("../")

dir_str = str(sys.argv[1])
nlayers = int(sys.argv[2])
c_init_order = int(sys.argv[3])
K = int(sys.argv[4])
sigma_init = float(sys.argv[5])
random_seed = int(sys.argv[6])

if (dir_str == 'Linear2D'):
    fixed_params = {"tau": 1.0}
    omega = 1
    mu = np.array([0.0, 2 * np.pi * omega])
    Sigma = np.array([1.0, 1.0])
    behavior = {"type": "oscillation", "means": mu, "variances": Sigma}
    system = Linear2D(fixed_params, behavior)

flow_type = "PlanarFlow"
mult_and_shift = "post"
arch_dict = {
    "D": system.D,
    "K": K,
    "flow_type": flow_type,
    "repeats": nlayers,
    "post_affine": True,
}

lr_order = -3

savedir = get_savedir(system, arch_dict, sigma_init, lr_order, c_init_order, random_seed, dir_str)
fname = savedir + 'opt_info.npz'
npzfile = np.load(fname)
Zs = npzfile['Zs']
log_q_zs = npzfile['log_q_zs']
Cs = npzfile['Cs']
alphas = npzfile['alphas']
check_rate = npzfile['check_rate']
epoch_inds = npzfile['epoch_inds']
dist_label_strs = system.z_labels
step = 1

start_time = time.time()
make_training_movie(Zs, log_q_zs, Cs, alphas, dist_label_strs, check_rate, epoch_inds, step)
end_time = time.time()

print('Took %.3f seconds to make the movie.' % (end_time - start_time))
