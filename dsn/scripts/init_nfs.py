from dsn.train_dsn import initialize_nf
import sys, os

os.chdir("../")

D = int(sys.argv[1]);
nlayers = int(sys.argv[2]);
sigma_init = float(sys.argv[3]);
random_seed = int(sys.argv[4]);

latent_dynamics = None;
TIF_flow_type = 'PlanarFlowLayer';
scale_layer = True;

flow_dict = {'latent_dynamics':latent_dynamics, \
             'TIF_flow_type':TIF_flow_type, \
             'repeats':nlayers, \
             'scale_layer':scale_layer};

initialize_nf(D, flow_dict, sigma_init, random_seed);