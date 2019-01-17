import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dsn.train_dsn import initialize_nf

D = 4
T = 1

latent_dynamics = None
TIF_flow_type = 'PlanarFlow'
nlayers = 10
elem_mult_flow = True

arch_dict = {'D':D, \
             'latent_dynamics':latent_dynamics, \
             'elem_mult_flow':elem_mult_flow, \
             'TIF_flow_type':TIF_flow_type, \
             'repeats':nlayers}

sigma_init = 1.0
random_seed = 1

initialize_nf(D, arch_dict, sigma_init, random_seed, min_iters=50000)
