from dsn.train_dsn import initialize_nf
import sys, os

os.chdir("../")

D = int(sys.argv[1])
num_masks = int(sys.argv[2])
nlayers = int(sys.argv[3])
upl = int(sys.argv[4])
sigma_init = float(sys.argv[5])
random_seed = int(sys.argv[6])

real_nvp_arch = {"num_masks": num_masks, "nlayers": nlayers, "upl": upl}

arch_dict = {
    "D": D,
    "mult_and_shift": None,
    "latent_dynamics": None,
    "TIF_flow_type": "RealNVP",
    "repeats": 1,
    "real_nvp_arch": real_nvp_arch,
}

min_iters = 50000

initialize_nf(D, arch_dict, sigma_init, random_seed, min_iters)
