# Copyright 2018 Sean Bittner, Columbia University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
import tensorflow as tf
import numpy as np
from tf_util.tf_util import count_layer_params
from tf_util.normalizing_flows import SoftPlusFlow
from tf_util.flows import SoftPlusLayer, IntervalFlowLayer
import scipy.stats
from scipy.special import gammaln, psi
import scipy.io as sio
from itertools import compress
from dsn.util.tf_DMFT_solvers import rank1_spont_chaotic_solve, \
                                     rank1_input_chaotic_solve, \
                                     rank2_CDD_chaotic_solve, \
                                     rank2_CDD_static_solve

DTYPE = tf.float64

def tile_for_conditions(tensor_list, num_conds):
    num_tensors = len(tensor_list)
    tiled = []
    for i in range(num_tensors):
        tensor_i = tensor_list[i]
        tiled.append(tf.tile(tensor_i, [1,num_conds]))
    return tiled

class system:
    """Base class for systems using DSN modeling.

    Degenerate solution networks (DSNs) learn the full parameter space of models
    given some model behavioral constraints.  Given some system choice and a 
    behavioral specification, these classes are designed to perform the system
    specific functions that are necessary for training the corresponding DSN.

    # Attributes
        self.D (int): Dimensionality of $$z$$.
        self.num_suff_stats (int): Dimensionality of behavioral constraint vector 
                                   $$T(x)$$.
        all_params (list): List of strings of all parameters of full system model.
        fixed_params (dict): Parameter string indexes its fixed value.
        free_params (list): List of strings in `all_params` but not `fixed_params.keys()`.
                            These params make up z.
        behavior (dict): Contains the behavioral type and the constraints.
        mu (np.array): The mean constrain vector for DSN optimization.
        all_param_labels (list): List of tex strings for all parameters.
        z_labels (list): List of tex strings for free parameters.
        T_x_labels (list): List of tex strings for elements of $$T(x)$$.
    """

    def __init__(self, fixed_params, behavior):
        """System constructor.

		# Arguments 
            fixed_params (dict): Specifies fixed parameters and their values.
			behavior (dict): Contains the behavioral type and the constraints.
	
		"""
        self.fixed_params = fixed_params
        self.behavior = behavior
        self.mu = self.compute_mu()
        self.all_params, self.all_param_labels = self.get_all_sys_params()
        self.free_params = self.get_free_params()
        self.z_labels = self.get_z_labels()
        self.T_x_labels = self.get_T_x_labels()
        self.D = len(self.z_labels)
        self.num_suff_stats = len(self.T_x_labels)
        self.behavior_str = self.get_behavior_str()
        self.support_mapping = None

    def get_all_sys_params(self,):
        """Returns ordered list of all system parameters and individual element labels.

        # Returns
            all_params (list): List of strings of all parameters of full system model.
            all_param_labels (list): List of tex strings for all parameters.
        """
        raise NotImplementedError()

    def get_free_params(self,):
        """Returns members of `all_params` not in `fixed_params.keys()`.

        # Returns
            free_params (list): List of strings of parameters in $$z$$.

        """
        free_params = []
        for param_str in self.all_params:
            if not param_str in self.fixed_params.keys():
                free_params.append(param_str)
        return free_params

    def get_z_labels(self,):
        """Returns `z_labels`.

        # Returns
            z_labels (list): List of tex strings for free parameters.

        """
        z_labels = []
        for free_param in self.free_params:
            z_labels += self.all_param_labels[free_param]
        return z_labels

    def get_behavior_str(self,):
        """Returns `behavior_str`.

        # Returns
            behavior_str (str): String for DSN filenaming.

        """
        raise NotImplementedError()

    def get_T_x_labels(self,):
        """Returns `T_x_labels`.

        # Returns
            T_x_labels (list): List of tex strings for elements of $$T(x)$$.

        """
        raise NotImplementedError()

    def compute_suff_stats(self, z):
        """Compute sufficient statistics of density network samples.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            T_x (tf.tensor): Sufficient statistics of samples.

        """
        raise NotImplementedError()

    def compute_mu(self,):
        """Calculate expected moment constraints given system paramterization.

        # Returns
            mu (np.array): Expected moment constraints.

        """
        raise NotImplementedError()

    def center_suff_stats_by_mu(self, T_x):
        """Center sufficient statistics by the mean parameters mu.
    
        # Arguments
            T_x (tf.tensor): Sufficient statistics of samples.
            mu (np.array): mean vector of constraints

        # Returns
            T_x_mu_centered (tf.tensor): Mean centered sufficient statistics of samples.
        
        """
        return T_x - np.expand_dims(np.expand_dims(self.mu, 0), 1)

    def get_behavior_str(self,):
        """Returns `behavior_str`.

        # Returns
            behavior_str (str): String for DSN filenaming.

        """
        type_str = self.behavior["type"]
        behavior_str = type_str + "_mu="
        for i in range(self.num_suff_stats):
            if (i > 0):
                behavior_str += "_"
            behavior_str += '%.2E' % self.mu[i]
        return behavior_str


class Linear2D(system):
    """Linear two-dimensional system.

    This is a simple system explored in the <a href="../#Linear2D_example">DSN tutorial</a>, which demonstrates the
    utility of DSNs in an intuitive way.  

    \\begin{equation}
    \\tau \dot{x} = Ax, A = \\begin{bmatrix} a_1 & a_2 \\\\\\\\ a_3 & a_4 \end{bmatrix}
    \end{equation}

    Behaviors:

    'oscillation' - specify a distribution of oscillatory frequencies

    # Attributes
        behavior (dict): see Linear2D.compute_suff_stats
    """

    def __init__(self, fixed_params, behavior):
        super().__init__(fixed_params, behavior)
        self.name = "Linear2D"
        self.support_mapping = None

    def get_all_sys_params(self,):
        """Returns ordered list of all system parameters and individual element labels.

         - $$A$$ - 2x2 dynamics matrix
         - $$\\tau$$ - scalar timescale parameter

        # Returns
            all_params (list): List of strings of all parameters of full system model.
            all_param_labels (list): List of tex strings for all parameters.
        """
        all_params = ["A", "tau"]
        all_param_labels = {
            "A": [r"$a_1$", r"$a_2$", r"$a_3$", r"$a_4$"],
            "tau": [r"$\tau$"],
        }
        return all_params, all_param_labels

    def get_T_x_labels(self,):
        """Returns `T_x_labels`.

        Behaviors:

        'oscillation' - $$[$$real($$\lambda_1$$), $$\\frac{\\text{imag}(\lambda_1)}{2 \pi}$$, real$$(\lambda_1)^2$$, $$(\\frac{\\text{imag}(\lambda_1)}{2 \pi})^2]$$

        # Returns
            T_x_labels (list): List of tex strings for elements of $$T(x)$$.

        """
        if self.behavior["type"] == "oscillation":
            T_x_labels = [
                r"real($\lambda_1$)",
                r"$\frac{imag(\lambda_1)}{2 \pi}$",
                r"real$(\lambda_1)^2$",
                r"$(\frac{imag(\lambda_1)}{2 \pi})^2$",
            ]
        else:
            raise NotImplementedError()
        return T_x_labels

    def compute_suff_stats(self, z):
        """Compute sufficient statistics of density network samples.

        Behaviors:

        'oscillation' - Specifies a distribution of oscillatory frequencies and 
                        expansion/decay factors using the eigendecomposition of
                        the dynamics matrix.
        \\begin{equation}
        E_{x\\sim p(x \\mid z)}\\left[T(x)\\right] = f_{p,T}(z) = E \\begin{bmatrix} \\text{real}(\\lambda_1) \\\\\\\\ \\frac{\\text{imag}(\\lambda_1)}{2\pi} \\\\\\\\ \\text{real}(\\lambda_1)^2 \\\\\\\\ (\\frac{\\text{imag}(\\lambda_1)}{2\pi}^2 \end{bmatrix}
        \end{equation}

		# Arguments
			z (tf.tensor): Density network system parameter samples.

		# Returns
			T_x (tf.tensor): Sufficient statistics of samples.

		"""
        if self.behavior["type"] == "oscillation":
            z_shape = tf.shape(z)
            K = z_shape[0]
            M = z_shape[1]

            # read free parameters from z vector
            ind = 0
            for free_param in self.free_params:
                if free_param == "A":
                    a1 = z[:, :, ind]
                    a2 = z[:, :, ind + 1]
                    a3 = z[:, :, ind + 2]
                    a4 = z[:, :, ind + 3]
                    ind += 4
                elif free_param == "tau":
                    tau = z[:, :, ind]
                    ind += 1

            # load fixed parameters
            for fixed_param in self.fixed_params.keys():
                if fixed_param == "A":
                    a1 = self.fixed_params["A"][0]
                    a2 = self.fixed_params["A"][1]
                    a3 = self.fixed_params["A"][2]
                    a4 = self.fixed_params["A"][3]
                elif fixed_param == "tau":
                    tau = self.fixed_params["tau"]

            # C = A / tau are the effective linear dynamics
            c1 = tf.divide(a1, tau)
            c2 = tf.divide(a2, tau)
            c3 = tf.divide(a3, tau)
            c4 = tf.divide(a4, tau)

            beta = tf.complex(
                tf.square(c1 + c4) - 4 * (c1 * c4 - c2 * c3), np.float64(0.0)
            )
            beta_sqrt = tf.sqrt(beta)
            real_common = tf.complex(0.5 * (c1 + c4), np.float64(0.0))

            lambda_1 = real_common + 0.5 * beta_sqrt
            lambda_1_real = tf.real(lambda_1)
            lambda_1_imag = tf.imag(lambda_1)
            moments = [
                lambda_1_real,
                lambda_1_imag,
                tf.square(lambda_1_real),
                tf.square(lambda_1_imag),
            ]
            T_x = tf.stack(moments, 2)
        else:
            raise NotImplementedError
        return T_x

    def compute_mu(self,):
        """Calculate expected moment constraints given system paramterization.

        # Returns
            mu (np.array): Expected moment constraints.

        """
        means = self.behavior["means"]
        variances = self.behavior["variances"]
        first_moments = means
        second_moments = np.square(means) + variances
        mu = np.concatenate((first_moments, second_moments), axis=0)
        return mu


class V1Circuit(system):
    """ 4-neuron V1 circuit.

        This is the standard 4-neuron rate model of V1 activity consisting of 
         - E: pyramidal (excitatory) neurons
         - P: parvalbumim expressing inhibitory neurons
         - S: somatostatin expressing inhibitory neurons
         - V: vasoactive intestinal peptide (VIP) expressing inhibitory neurons

         [include a graphic of the circuit connectivity]

        The dynamics of each neural populations average rate 
        $$r = \\begin{bmatrix} r_E \\\\ r_P \\\\ r_S \\\\ r_V \end{bmatrix}$$
        are given by:
        \\begin{equation}
        \\tau \\frac{dr}{dt} = -r + [Wr + h]_+^n
        \end{equation}

        In some cases, these neuron types do not send projections to one of the other 
        types.  Additionally, much work, such as ([Pfeffer et al. 2013](#Pfeffer2013Inhibition)) 
        has been done to measure the relative magnitudes of the synaptic projections 
        between neural types.
        \\begin{equation}
        W = \\begin{bmatrix} W_{EE} & -1.0 & -0.54 & 0 \\\\\\\\ W_{PE} & -1.01 & -0.33 & 0 \\\\\\\\ W_{SE} & 0 & 0 & -0.15 \\\\\\\\ W_{VE} & -0.22 & -0.77 & 0 \end{bmatrix}
        \end{equation}

        In this model, we are interested in capturing V1 responses across varying 
        contrasts $$c$$, stimulus sizes $$s$$, and locomotion $$r$$ conditions as
        in ([Dipoppa et al. 2018](#Dipoppa2018Vision)).

        \\begin{equation} 
        h = b + g_{FF}(c) h_{FF} + g_{LAT}(c,s) h_{LAT} + g_{RUN}(r) h_{RUN}
        \end{equation}

        \\begin{equation} \\begin{bmatrix} h_E \\\\\\\\ h_P \\\\\\\\ h_S \\\\\\\\ h_V \end{bmatrix}
         = \\begin{bmatrix} b_E \\\\\\\\ b_P \\\\\\\\ b_S \\\\\\\\ b_V \end{bmatrix} + g_{FF}(c) \\begin{bmatrix} h_{FF,E} \\\\\\\\ h_{FF,P} \\\\\\\\ 0 \\\\\\\\ 0 \end{bmatrix} + g_{LAT}(c,s) \\begin{bmatrix} h_{LAT,E} \\\\\\\\ h_{LAT,P} \\\\\\\\ h_{LAT,S} \\\\\\\\ h_{LAT,V} \end{bmatrix} + g_{RUN}(r) \\begin{bmatrix} h_{RUN,E} \\\\\\\\ h_{RUN,P} \\\\\\\\ h_{RUN,S} \\\\\\\\ h_{RUN,V} \end{bmatrix}
        \end{equation}

        where $$g_{FF}(c)$$, $$g_{LAT}(c,s)$$, and $$g_{FF}(r)$$ modulate the input
        parmeterization $$h$$ according to condition.  See initialization argument
        `model_opts` on how to set the form of these functions. 

    # Attributes
        behavior (dict): see V1Circuit.compute_suff_stats
        model_opts (dict): 
          * model_opts[`'g_FF'`] 
            * `'c'` (default) $$g_{FF}(c) = c$$ 
            * `'saturate'` $$g_{FF}(c) = \\frac{c^a}{c_{50}^a + c^a}$$
          * model_opts[`'g_LAT'`] 
            * `'linear'` (default) $$g_{LAT}(c,s) = c[s_0 - s]_+$$ 
            * `'square'` $$g_{LAT}(c,s) = c[s_0^2 - s^2]_+$$
          * model_opts[`'g_RUN'`] 
            * `'r'` (default) $$g_{RUN}(r) = r$$ 
        T (int): Number of simulation time points.
        dt (float): Time resolution of simulation.
        init_conds (list): Specifies the initial state of the system.
    """

    def __init__(
        self,
        fixed_params,
        behavior,
        model_opts={"g_FF": "c", "g_LAT": "linear", "g_RUN": "r"},
        T=40,
        dt=0.25,
        init_conds=np.expand_dims(np.array([1.0, 1.1, 1.2, 1.3]), 1),
    ):
        self.model_opts = model_opts
        super().__init__(fixed_params, behavior)
        self.name = "V1Circuit"
        self.T = T
        self.dt = dt
        self.init_conds = init_conds

        # compute number of conditions C
        num_c = self.behavior["c_vals"].shape[0]
        num_s = self.behavior["s_vals"].shape[0]
        num_r = self.behavior["r_vals"].shape[0]
        self.C = num_c * num_s * num_r

    def get_all_sys_params(self,):
        """Returns ordered list of all system parameters and individual element labels.

         - $$W_{EE}$$ - strength of excitatory-to-excitatory projection
         - $$W_{PE}$$ - strength of excitatory-to-parvalbumin projection 
         - $$W_{SE}$$ - strength of excitatory-to-somatostatin projection 
         - $$W_{VE}$$ - strength of excitatory-to-VIP projection
         - $$b_{E}$$ - constant input to excitatory population 
         - $$b_{P}$$ - constant input to parvalbumin population 
         - $$b_{S}$$ - constant input to somatostatin population 
         - $$b_{V}$$ - constant input to VIP population 
         - $$h_{FF,E}$$ - feed-forward input to excitatory population
         - $$h_{FF,P}$$ - feed-forward input to parvalbumin population
         - $$h_{LAT,E}$$ - lateral input to excitatory population
         - $$h_{LAT,P}$$ - lateral input to parvalbumin population
         - $$h_{LAT,S}$$ - lateral input to somatostatin population
         - $$h_{LAT,V}$$ - lateral input to VIP population
         - $$h_{RUN,E}$$ - locomotion input to excitatory population
         - $$h_{RUN,P}$$ - locomotion input to parvalbumin population
         - $$h_{RUN,S}$$ - locomotion input to somatostatin population
         - $$h_{RUN,V}$$ - locomotion input to VIP population
         - $$\\tau$$ - dynamics timescale
         - $$n$$ - scalar for power of dynamics
         - $$s_0$$ - reference stimulus level

         When `model_opts['g_FF'] == 'saturate'`
         - $$a$$ - contrast saturation shape
         - $$c_{50}$$ - constrast at 50%

        # Returns
            all_params (list): List of strings of all parameters of full system model.
            all_param_labels (list): List of tex strings for all parameters.
        """
        all_params = [
            "W_EE",
            "W_PE",
            "W_SE",
            "W_VE",
            "b_E",
            "b_P",
            "b_S",
            "b_V",
            "h_FFE",
            "h_FFP",
            "h_LATE",
            "h_LATP",
            "h_LATS",
            "h_LATV",
            "h_RUNE",
            "h_RUNP",
            "h_RUNS",
            "h_RUNV",
            "tau",
            "n",
            "s_0",
        ]
        all_param_labels = {
            "W_EE": [r"$W_{EE}$"],
            "W_PE": [r"$W_{PE}$"],
            "W_SE": [r"$W_{SE}$"],
            "W_VE": [r"$W_{VE}$"],
            "b_E": [r"$b_{E}$"],
            "b_P": [r"$b_{P}$"],
            "b_S": [r"$b_{S}$"],
            "b_V": [r"$b_{V}$"],
            "h_FFE": [r"$h_{FF,E}$"],
            "h_FFP": [r"$h_{FF,P}$"],
            "h_LATE": [r"$h_{LAT,E}$"],
            "h_LATP": [r"$h_{LAT,P}$"],
            "h_LATS": [r"$h_{LAT,S}$"],
            "h_LATV": [r"$h_{LAT,V}$"],
            "h_RUNE": [r"$h_{RUN,E}$"],
            "h_RUNP": [r"$h_{RUN,P}$"],
            "h_RUNS": [r"$h_{RUN,S}$"],
            "h_RUNV": [r"$h_{RUN,V}$"],
            "tau": [r"$\tau$"],
            "n": [r"$n$"],
            "s_0": [r"$s_0$"],
        }

        if self.model_opts["g_FF"] == "saturate":
            all_params += ["a", "c_50"]
            all_param_labels.update({"a": r"$a$", "c_50": r"$c_{50}$"})

        return all_params, all_param_labels

    def get_T_x_labels(self,):
        """Returns `T_x_labels`.

        Behaviors:

        'difference' - $$[d_{E,ss}, d_{P,ss}, d_{S,ss}, d_{V,ss}, d_{E,ss}^2, d_{P,ss}^2, d_{S,ss}^2, d_{V,ss}^2]$$
        
        'data' - $$[r_{E,ss}(c,s,r), ...,  r_{E,ss}(c,s,r)^2, ...]$$

        # Returns
            T_x_labels (list): List of tex strings for elements of $$T(x)$$.

        """
        if self.behavior["type"] == "difference":
            all_T_x_labels = [
                r"$d_{E,ss}$",
                r"$d_{P,ss}$",
                r"$d_{S,ss}$",
                r"$d_{V,ss}$",
                r"$d_{E,ss}^2$",
                r"$d_{P,ss}^2$",
                r"$d_{S,ss}^2$",
                r"$d_{V,ss}^2$",
            ]
            diff_inds = self.behavior["diff_inds"]
            label_inds = diff_inds + list(map(lambda x: x + 4, diff_inds))
            T_x_labels = []
            for i in range(len(label_inds)):
                T_x_labels.append(all_T_x_labels[label_inds[i]])
        else:
            raise NotImplementedError()
        return T_x_labels

    def filter_Z(self, z):
        """Returns the system matrix/vector variables depending free parameter ordering.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            W (tf.tensor): [C,M,4,4] Dynamics matrices.
            b (tf.tensor): [1,M,4,1] Static inputs.
            h_FF (tf.tensor): [1,M,4,1] Feed forward inputs.
            h_LAT (tf.tensor): [1,M,4,1] Lateral inputs.
            h_RUN (tf.tensor): [1,M,4,1] Running inputs.
            tau (tf.tensor): [C,M,1,1] Dynamics timescales.
            n (tf.tensor): [C,M,1,1] Dynamics power coefficients.
            s_0 (tf.tensor): [1,M,1,1] Reference stimulus values.
            a (tf.tensor): [1,M,1,1] Contrast saturation shape.
            c_50 (tf.tensor): [1,M,1,1] Contrast at 50%.

        """
        z_shape = tf.shape(z)
        K = z_shape[0]
        M = z_shape[1]

        # Assumed parameters
        W_EP = 1.0 * tf.ones((self.C, M), dtype=DTYPE)
        W_ES = 0.54 * tf.ones((self.C, M), dtype=DTYPE)

        W_PP = 1.01 * tf.ones((self.C, M), dtype=DTYPE)
        W_PS = 0.33 * tf.ones((self.C, M), dtype=DTYPE)

        W_SV = 0.15 * tf.ones((self.C, M), dtype=DTYPE)

        W_VP = 0.22 * tf.ones((self.C, M), dtype=DTYPE)
        W_VS = 0.77 * tf.ones((self.C, M), dtype=DTYPE)

        # read free parameters from z vector
        ind = 0
        for free_param in self.free_params:
            if free_param == "W_EE":
                W_EE = tf.tile(z[:, :, ind], [self.C, 1])
            elif free_param == "W_PE":
                W_PE = tf.tile(z[:, :, ind], [self.C, 1])
            elif free_param == "W_SE":
                W_SE = tf.tile(z[:, :, ind], [self.C, 1])
            elif free_param == "W_VE":
                W_VE = tf.tile(z[:, :, ind], [self.C, 1])

            elif free_param == "b_E":
                b_E = tf.tile(z[:, :, ind], [1, 1])
            elif free_param == "b_P":
                b_P = tf.tile(z[:, :, ind], [1, 1])
            elif free_param == "b_S":
                b_S = tf.tile(z[:, :, ind], [1, 1])
            elif free_param == "b_V":
                b_V = tf.tile(z[:, :, ind], [1, 1])

            elif free_param == "h_FFE":
                h_FFE = tf.tile(z[:, :, ind], [1, 1])
            elif free_param == "h_FFP":
                h_FFP = tf.tile(z[:, :, ind], [1, 1])

            elif free_param == "h_LATE":
                h_LATE = tf.tile(z[:, :, ind], [1, 1])
            elif free_param == "h_LATP":
                h_LATP = tf.tile(z[:, :, ind], [1, 1])
            elif free_param == "h_LATS":
                h_LATS = tf.tile(z[:, :, ind], [1, 1])
            elif free_param == "h_LATV":
                h_LATV = tf.tile(z[:, :, ind], [1, 1])

            elif free_param == "h_RUNE":
                h_RUNE = tf.tile(z[:, :, ind], [1, 1])
            elif free_param == "h_RUNP":
                h_RUNP = tf.tile(z[:, :, ind], [1, 1])
            elif free_param == "h_RUNS":
                h_RUNS = tf.tile(z[:, :, ind], [1, 1])
            elif free_param == "h_RUNV":
                h_RUNV = tf.tile(z[:, :, ind], [1, 1])

            elif free_param == "tau":
                tau = tf.tile(z[:, :, ind], [self.C, 1])
            elif free_param == "n":
                n = tf.tile(z[:, :, ind], [self.C, 1])
            elif free_param == "s_0":
                s_0 = tf.tile(z[:, :, ind], [1, 1])
            elif free_param == "a":
                a = tf.tile(z[:, :, ind], [1, 1])
            elif free_param == "c_50":
                c_50 = tf.tile(z[:, :, ind], [1, 1])

            else:
                print("Error: unknown free parameter: %s." % free_param)
                raise NotImplementedError()
            ind += 1

        # load fixed parameters
        for fixed_param in self.fixed_params.keys():
            if fixed_param == "W_EE":
                W_EE = self.fixed_params[fixed_param] * tf.ones(
                    (self.C, M), dtype=DTYPE
                )
            elif fixed_param == "W_PE":
                W_PE = self.fixed_params[fixed_param] * tf.ones(
                    (self.C, M), dtype=DTYPE
                )
            elif fixed_param == "W_SE":
                W_SE = self.fixed_params[fixed_param] * tf.ones(
                    (self.C, M), dtype=DTYPE
                )
            elif fixed_param == "W_VE":
                W_VE = self.fixed_params[fixed_param] * tf.ones(
                    (self.C, M), dtype=DTYPE
                )

            elif fixed_param == "b_E":
                b_E = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
            elif fixed_param == "b_P":
                b_P = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
            elif fixed_param == "b_S":
                b_S = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
            elif fixed_param == "b_V":
                b_V = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)

            elif fixed_param == "h_FFE":
                h_FFE = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
            elif fixed_param == "h_FFP":
                h_FFP = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)

            elif fixed_param == "h_LATE":
                h_LATE = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
            elif fixed_param == "h_LATP":
                h_LATP = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
            elif fixed_param == "h_LATS":
                h_LATS = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
            elif fixed_param == "h_LATV":
                h_LATV = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)

            elif fixed_param == "h_RUNE":
                h_RUNE = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
            elif fixed_param == "h_RUNP":
                h_RUNP = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
            elif fixed_param == "h_RUNS":
                h_RUNS = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
            elif fixed_param == "h_RUNV":
                h_RUNV = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)

            elif fixed_param == "tau":
                tau = self.fixed_params[fixed_param] * tf.ones((self.C, M), dtype=DTYPE)
            elif fixed_param == "n":
                n = self.fixed_params[fixed_param] * tf.ones((self.C, M), dtype=DTYPE)
            elif fixed_param == "s_0":
                s_0 = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
            elif fixed_param == "a":
                a = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
            elif fixed_param == "c_50":
                c_50 = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)

            else:
                print("Error: unknown fixed parameter: %s." % fixed_param)
                raise NotImplementedError()

        # Gather weights into the dynamics matrix W [C,M,4,4]
        W_EX = tf.stack(
            [W_EE, -W_EP, -W_ES, tf.zeros((self.C, M), dtype=DTYPE)], axis=2
        )
        W_PX = tf.stack(
            [W_PE, -W_PP, -W_PS, tf.zeros((self.C, M), dtype=DTYPE)], axis=2
        )
        W_SX = tf.stack(
            [
                W_SE,
                tf.zeros((self.C, M), dtype=DTYPE),
                tf.zeros((self.C, M), dtype=DTYPE),
                -W_SV,
            ],
            axis=2,
        )
        W_VX = tf.stack(
            [W_VE, -W_VP, -W_VS, tf.zeros((self.C, M), dtype=DTYPE)], axis=2
        )
        W = tf.stack([W_EX, W_PX, W_SX, W_VX], axis=2)

        # Gather inputs into b [K,M,4,1]
        b = tf.expand_dims(tf.stack([b_E, b_P, b_S, b_V], axis=2), 3)
        h_FF = tf.expand_dims(
            tf.stack(
                [
                    h_FFE,
                    h_FFP,
                    tf.zeros((1, M), dtype=DTYPE),
                    tf.zeros((1, M), dtype=DTYPE),
                ],
                axis=2,
            ),
            3,
        )
        h_LAT = tf.expand_dims(tf.stack([h_LATE, h_LATP, h_LATS, h_LATV], axis=2), 3)
        h_RUN = tf.expand_dims(tf.stack([h_RUNE, h_RUNP, h_RUNS, h_RUNV], axis=2), 3)

        # tau [K,M,1,1]
        tau = tf.expand_dims(tf.expand_dims(tau, 2), 3)
        # dynamics power [K,M,1,1]
        n = tf.expand_dims(tf.expand_dims(n, 2), 3)
        # reference stimulus [K,M,1,1]
        s_0 = tf.expand_dims(tf.expand_dims(s_0, 2), 3)

        if self.model_opts["g_LAT"] == "saturate":
            # saturation shape [K,M,1,1]
            a = tf.expand_dims(tf.expand_dims(a, 2), 3)
            # 50% constrast value [K,M,1,1]
            c_50 = tf.expand_dims(tf.expand_dims(c_50, 2), 3)
        else:
            a = None
            c_50 = None

        return W, b, h_FF, h_LAT, h_RUN, tau, n, s_0, a, c_50

    def compute_h(self, b, h_FF, h_LAT, h_RUN, s_0, a=None, c_50=None):
        num_c = self.behavior["c_vals"].shape[0]
        num_s = self.behavior["s_vals"].shape[0]
        num_r = self.behavior["r_vals"].shape[0]
        hs = []
        for i in range(num_c):
            c = self.behavior["c_vals"][i]
            # compute g_FF for this condition
            if self.model_opts["g_FF"] == "c":
                g_FF = c
            elif self.model_opts["g_FF"] == "saturate":
                g_FF = tf.divide(tf.pow(c, a), tf.pow(c_50, a) + tf.pow(c, a))
            else:
                raise NotImplementedError()

            for j in range(num_s):
                s = self.behavior["s_vals"][j]
                # compute g_LAT for this condition
                if self.model_opts["g_LAT"] == "linear":
                    g_LAT = tf.multiply(c, tf.nn.relu(s - s_0))
                elif self.model_opts["g_LAT"] == "square":
                    g_LAT = tf.multiply(c, tf.nn.relu(tf.square(s) - tf.square(s_0)))
                else:
                    raise NotImplementedError()

                for k in range(num_r):
                    if self.model_opts["g_RUN"] == "r":
                        r = self.behavior["r_vals"][k]
                    else:
                        raise NotImplementedError()

                    g_RUN = r
                    h_csr = (
                        b
                        + tf.multiply(g_FF, h_FF)
                        + tf.multiply(g_LAT, h_LAT)
                        + tf.multiply(g_RUN, h_RUN)
                    )
                    hs.append(h_csr)
        h = tf.concat(hs, axis=0)

        return h

    def simulate(self, z):
        """Simulate the V1 4-neuron circuit given parameters z.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            g(z) (tf.tensor): Simulated system activity.

        """

        # get number of batch samples
        z_shape = tf.shape(z)
        K = z_shape[0]
        M = z_shape[1]

        W, b, h_FF, h_LAT, h_RUN, tau, n, s_0, a, c_50 = self.filter_Z(z)
        h = self.compute_h(b, h_FF, h_LAT, h_RUN, s_0, a, c_50)

        # initial conditions
        r0 = tf.constant(
            np.expand_dims(np.expand_dims(self.init_conds, 0), 0), dtype=DTYPE
        )
        r0 = tf.tile(r0, [self.C, M, 1, 1])
        # [K,M,4,1]

        # construct the input
        def f(r, t):
            drdt = tf.divide(-r + tf.pow(tf.nn.relu(tf.matmul(W, r) + h), n), tau)
            return tf.clip_by_value(drdt, -1e30, 1e30)

        # worst-case cost is about
        # time = dt*T = 10 
        # r_ss = 1e30*time = 1e31
        # cost second mom term
        # r_ss2 = r_ss**2 = 1e62
        # in l2 norm over 1000 batch
        # cost ~~ 1e3*r_ss2**2 = 1e124*1e3 = 1e127

        # bound should be 1e308
        # going to 1e45 doesnt work for some reason?


        # time axis
        t = np.arange(0, self.T * self.dt, self.dt)

        # simulate ODE
        r_t = tf.contrib.integrate.odeint_fixed(f, r0, t, method="rk4")
        return r_t

    def compute_suff_stats(self, z):
        """Compute sufficient statistics of density network samples.

        Behaviors:

        'difference' - 

          The total number of conditions from all of 
          self,behavior.c_vals, s_vals, and r_vals should be two.  
          The steady state of the first condition $$(c_1,s_1,r_1)$$ is 
          subtracted from that of the second condition $$(c_2,s_2,r_2)$$ to get a 
          difference vector
          \\begin{equation}
          d_{\\alpha,ss} = r_{\\alpha,ss}(c_2,s_2,r_2) - r_{\\alpha,ss}(c_1,s_1,r_1)
          \end{equation}
        
          The total constraint vector is
          \\begin{equation}
          E_{x\\sim p(x \\mid z)}\\left[T(x)\\right] = \\begin{bmatrix} d_{E,ss} \\\\\\\\ d_{P,ss} \\\\\\\\ d_{S,ss} \\\\\\\\ d_{V,ss} \\\\\\\\ d_{E,ss}^2 \\\\\\\\ d_{P,ss}^2 \\\\\\\\ d_{S,ss}^2 \\\\\\\\ d_{V,ss}^2 \end{bmatrix}
          \end{equation}

        
        'data' - 

          The user specifies the grid inputs for conditions via 
          self.behavior.c_vals, s_vals, and r_vals.  The first and second
          moments of the steady states for these conditions make up the
          sufficient statistics vector.  Since the index is $$(c,s,r)$$, 
          values of r are iterated over first, then s, then c (as is 
          the c-standard) to construct the $$T(x)$$ vector.

          The total constraint vector is
          \\begin{equation}
          E_{x\\sim p(x \\mid z)}\\left[T(x)\\right] = \\begin{bmatrix} r_{E,ss}(c,s,r) \\\\\\\\ ... \\\\\\\\  r_{E,ss}(c,s,r)^2 \\\\\\\\ ... \end{bmatrix}
          \end{equation}

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            T_x (tf.tensor): Sufficient statistics of samples.

        """

        if self.behavior["type"] in ["data", "difference"]:
            T_x = self.simulation_suff_stats(z)
        else:
            raise NotImplementedError()

        return T_x

    def simulation_suff_stats(self, z):
        """Compute sufficient statistics that require simulation.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            T_x (tf.tensor): Simulation-derived sufficient statistics of samples.

        """

        # r1_t, r2_t = self.simulate(z);
        r_t = self.simulate(z)
        # [T, C, M, D, 1]

        if self.behavior["type"] == "difference":
            diff_inds = self.behavior["diff_inds"]
            r1_ss_list = []
            r2_ss_list = []
            for ind in diff_inds:
                r1_ss_list.append(r_t[-1, 0, :, ind, 0])
                r2_ss_list.append(r_t[-1, 1, :, ind, 0])
            r1_ss = tf.stack(r1_ss_list, axis=1)
            r2_ss = tf.stack(r2_ss_list, axis=1)
            diff_ss = tf.expand_dims(r2_ss - r1_ss, 0)
            T_x = tf.concat((diff_ss, tf.square(diff_ss)), 2)

        elif self.behavior["type"] == "data":
            r_shape = tf.shape(r_t)
            M = tf.shape[2]
            r_ss = tf.transpose(r1_t[-1, :, :, :, 0], [1, 2, 0])  # [M,C,D];
            r_ss = tf.reshape(r_ss, [M, self.C * self.D])
            T_x = tf.concat((r_ss, tf.square(r_ss)), 2)

        return T_x

    def compute_mu(self,):
        """Calculate expected moment constraints given system paramterization.

        # Returns
            mu (np.array): Expected moment constraints.

        """

        if self.behavior["type"] == "difference":
            means = self.behavior["d_mean"]
            variances = self.behavior["d_var"]
        elif self.behavior["type"] == "data":
            means = np.reshape(self.behavior["r_mean"], (self.C * self.D,))
            variances = np.reshape(self.behavior["r_var"], (self.C * self.D,))
        first_moments = means
        second_moments = np.square(means) + variances
        mu = np.concatenate((first_moments, second_moments), axis=0)
        return mu

    def support_mapping(self, inputs):
        """Maps from real numbers to support of parameters.

        # Arguments:
            inputs (np.array): Input from previous layers of the DSN.

        # Returns
            Z (np.array): Samples from the DSN at the final layer.
        """
        return SoftPlusFlow([], inputs)


class SCCircuit(system):
    """ 4-neuron SC circuit.

        This is a 4-neuron rate model of SC activity across two hemispheres
         - LP: Left, Pro
         - LA: Left, Anti
         - RA: Right, Anti
         - RP: Right, Pro

         [include a graphic of the circuit connectivity]

         [add equations]

    # Attributes
        behavior (dict): see SCCircuit.compute_suff_stats
    """

    def __init__(
        self,
        fixed_params,
        behavior,
        model_opts={"params":"reduced"}
    ):
        self.model_opts = model_opts
        super().__init__(fixed_params, behavior)
        self.name = "SCCircuit"

        # time course for task
        self.t_cue_delay = 1.2
        self.t_choice = 0.6
        t_total = self.t_cue_delay + self.t_choice
        self.dt = 0.024
        self.t = np.arange(0.0, t_total, self.dt)
        self.T = self.t.shape[0]

        # number of frozen noises to average over
        self.N = 100 
        # Sample frozen noise.
        # Rates are stored as (T, C, M, 4, N).
        # C and M are broadcast dimensions.
        self.w = np.random.normal(0.0, 1.0, (self.T,1,1,4,self.N))

        if (behavior["type"] in ["standard", "means", "pvar", "feasible"]):
            self.C = 1

    def get_all_sys_params(self,):
        """Returns ordered list of all system parameters and individual element labels.

         - $$sW$$ - strength of self connections
         - $$vW$$ - strength of vertical connections
         - $$dW$$ - strength of diagonal connections
         - $$hW$$ - strength of horizontal connections
         - $$E_constant$$ - constant input
         - $$E_Pbias$$ - bias input to Pro units
         - $$E_Prule$$ - input to Pro units in Pro condition
         - $$E_Arule$$ - input to Anti units in Anti condition
         - $$E_choice$$ - input during choice period
         - $$E_light$$ - input due to light stimulus


        # Returns
            all_params (list): List of strings of all parameters of full system model.
            all_param_labels (list): List of tex strings for all parameters.
        """
        if (self.model_opts["params"] == "full"):
            all_params = [
                "sW_P",
                "sW_A",
                "vW_PA",
                "vW_AP",
                "dW_PA",
                "dW_AP",
                "hW_P",
                "hW_A",
                "E_constant",
                "E_Pbias",
                "E_Prule",
                "E_Arule",
                "E_choice",
                "E_light",
            ]
            all_param_labels = {
                "sW_P": [r"$sW_{P}$"],
                "sW_A": [r"$sW_{A}$"],
                "vW_PA": [r"$vW_{PA}$"],
                "vW_AP": [r"$vW_{AP}$"],
                "dW_PA": [r"$dW_{PA}$"],
                "dW_AP": [r"$dW_{AP}$"],
                "hW_P": [r"$hW_{P}$"],
                "hW_A": [r"$hW_{A}$"],
                "E_constant": [r"$E_{constant}$"],
                "E_Pbias": [r"$E_{P,bias}$"],
                "E_Prule": [r"$E_{P,rule}$"],
                "E_Arule": [r"$E_{A,rule}$"],
                "E_choice": [r"$E_{choice}$"],
                "E_light": [r"$E_{light}$"],
            }
        elif (self.model_opts["params"] == "reduced"):
            all_params = [
                "sW",
                "vW",
                "dW",
                "hW",
                "E_constant",
                "E_Pbias",
                "E_Prule",
                "E_Arule",
                "E_choice",
                "E_light",
            ]
            all_param_labels = {
                "sW": [r"$sW$"],
                "vW": [r"$vW$"],
                "dW": [r"$dW$"],
                "hW": [r"$hW$"],
                "E_constant": [r"$E_{constant}$"],
                "E_Pbias": [r"$E_{P,bias}$"],
                "E_Prule": [r"$E_{P,rule}$"],
                "E_Arule": [r"$E_{A,rule}$"],
                "E_choice": [r"$E_{choice}$"],
                "E_light": [r"$E_{light}$"],
            }
        else:
            raise NotImplementedError()

        return all_params, all_param_labels

    def get_T_x_labels(self,):
        """Returns `T_x_labels`.

        Behaviors:

        # Returns
            T_x_labels (list): List of tex strings for elements of $$T(x)$$.

        """
        if self.behavior["type"] == "standard":
            T_x_labels = [
                r"$E_{\partial W}[{V_{LP},L}]$",
                r"$Var_{\partial W}[{V_{LP},L}] - p(1-p)$",
                r"$E_{\partial W}[ {V_{LP},L}]^2$",
                r"$(Var_{\partial W}[ {V_{LP},L}] - p(1-p))^2$",
            ]
        elif self.behavior["type"] == "means":
            T_x_labels = [
                r"$E_{\partial W}[{V_{LP},L}]$",
                r"$Var_{\partial W}[{V_{LP},L}] - p(1-p)$"
            ]
        elif self.behavior["type"] == "pvar":
            T_x_labels = [
                r"$E_{\partial W}[{V_{LP},L}]$",
                r"$Var_{\partial W}[{V_{LP},L}] - p(1-p)$",
                r"$E_{\partial W}[ {V_{LP},L}]^2$",
            ]
        elif self.behavior["type"] == "feasible":
            T_x_labels = [
                r"$Var_{\partial W}[{V_{LP},L}]$",
                r"$Var_{\partial W}[ {V_{LP},L}]^2$",
            ]
        else:
            raise NotImplementedError()
        return T_x_labels

    def filter_Z(self, z):
        """Returns the system matrix/vector variables depending free parameter ordering.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            W (tf.tensor): [C,M,4,4] Dynamics matrices.
            I (tf.tensor): [C,M,4,1] Static inputs.

        """
        z_shape = tf.shape(z)
        K = z_shape[0]
        M = z_shape[1]

        # read free parameters from z vector
        if (self.model_opts["params"] == "full"):
            ind = 0
            for free_param in self.free_params:
                if free_param == "sW_P":
                    sW_P = tf.tile(z[:, :, ind], [self.C, 1])
                elif free_param == "sW_A":
                    sW_A = tf.tile(z[:, :, ind], [self.C, 1])
                elif free_param == "vW_PA":
                    vW_PA = tf.tile(z[:, :, ind], [self.C, 1])
                elif free_param == "vW_AP":
                    vW_AP = tf.tile(z[:, :, ind], [self.C, 1])
                elif free_param == "dW_PA":
                    dW_PA = tf.tile(z[:, :, ind], [self.C, 1])
                elif free_param == "dW_AP":
                    dW_AP = tf.tile(z[:, :, ind], [self.C, 1])
                elif free_param == "hW_P":
                    hW_P = tf.tile(z[:, :, ind], [self.C, 1])
                elif free_param == "hW_A":
                    hW_A = tf.tile(z[:, :, ind], [self.C, 1])
                elif free_param == "E_constant":
                    E_constant = tf.expand_dims(tf.expand_dims(tf.expand_dims(z[:, :, ind], 1), 3), 4)
                elif free_param == "E_Pbias":
                    E_Pbias = tf.expand_dims(tf.expand_dims(tf.expand_dims(z[:, :, ind], 1), 3), 4)
                elif free_param == "E_Prule":
                    E_Prule = tf.expand_dims(tf.expand_dims(tf.expand_dims(z[:, :, ind], 1), 3), 4)
                elif free_param == "E_Arule":
                    E_Arule = tf.expand_dims(tf.expand_dims(tf.expand_dims(z[:, :, ind], 1), 3), 4)
                elif free_param == "E_choice":
                    E_choice = tf.expand_dims(tf.expand_dims(tf.expand_dims(z[:, :, ind], 1), 3), 4)
                elif free_param == "E_light":
                    E_light = tf.expand_dims(tf.expand_dims(tf.expand_dims(z[:, :, ind], 1), 3), 4)

                else:
                    print("Error: unknown free parameter: %s." % free_param)
                    raise NotImplementedError()
                ind += 1

            # load fixed parameters
            for fixed_param in self.fixed_params.keys():
                if fixed_param == "sW_P":
                    sW_P = self.fixed_params[fixed_param] * tf.ones(
                        (self.C, M), dtype=DTYPE
                    )
                elif fixed_param == "sW_A":
                    sW_A = self.fixed_params[fixed_param] * tf.ones(
                        (self.C, M), dtype=DTYPE
                    )
                elif fixed_param == "vW_PA":
                    vW_PA = self.fixed_params[fixed_param] * tf.ones(
                        (self.C, M), dtype=DTYPE
                    )
                elif fixed_param == "vW_AP":
                    vW_AP = self.fixed_params[fixed_param] * tf.ones(
                        (self.C, M), dtype=DTYPE
                    )
                elif fixed_param == "dW_PA":
                    dW_PA = self.fixed_params[fixed_param] * tf.ones(
                        (self.C, M), dtype=DTYPE
                    )
                elif fixed_param == "dW_AP":
                    dW_AP = self.fixed_params[fixed_param] * tf.ones(
                        (self.C, M), dtype=DTYPE
                    )
                elif fixed_param == "hW_P":
                    hW_P = self.fixed_params[fixed_param] * tf.ones(
                        (self.C, M), dtype=DTYPE
                    )
                elif fixed_param == "hW_A":
                    hW_A = self.fixed_params[fixed_param] * tf.ones(
                        (self.C, M), dtype=DTYPE
                    )
                elif fixed_param == "E_constant":
                    E_constant = self.fixed_params[fixed_param] * tf.ones((1, 1, M, 1, 1), dtype=DTYPE)
                elif fixed_param == "E_Pbias":
                    E_Pbias = self.fixed_params[fixed_param] * tf.ones((1, 1, M, 1, 1), dtype=DTYPE)
                elif fixed_param == "E_Prule":
                    E_Prule = self.fixed_params[fixed_param] * tf.ones((1, 1, M, 1, 1), dtype=DTYPE)
                elif fixed_param == "E_Arule":
                    E_Arule = self.fixed_params[fixed_param] * tf.ones((1, 1, M, 1, 1), dtype=DTYPE)
                elif fixed_param == "E_choice":
                    E_choice = self.fixed_params[fixed_param] * tf.ones((1, 1, M, 1, 1), dtype=DTYPE)
                elif fixed_param == "E_light":
                    E_light = self.fixed_params[fixed_param] * tf.ones((1, 1, M, 1, 1), dtype=DTYPE)

                else:
                    print("Error: unknown fixed parameter: %s." % fixed_param)
                    raise NotImplementedError()

            # Gather weights into the dynamics matrix W [C,M,4,4]
            Wrow1 = tf.stack([sW_P,  vW_PA, dW_PA, hW_P], axis=2)
            Wrow2 = tf.stack([vW_AP, sW_A,  hW_A,  dW_AP], axis=2)
            Wrow3 = tf.stack([dW_AP, hW_P,  sW_A,  vW_AP], axis=2)
            Wrow4 = tf.stack([hW_A,  dW_PA, vW_PA, sW_P], axis=2)
            W = tf.stack([Wrow1, Wrow2, Wrow3, Wrow4], axis=2)
        elif (self.model_opts["params"] == "reduced"):
            ind = 0
            for free_param in self.free_params:
                if free_param == "sW":
                    sW = tf.tile(z[:, :, ind], [self.C, 1])
                elif free_param == "vW":
                    vW = tf.tile(z[:, :, ind], [self.C, 1])
                elif free_param == "dW":
                    dW = tf.tile(z[:, :, ind], [self.C, 1])
                elif free_param == "hW":
                    hW = tf.tile(z[:, :, ind], [self.C, 1])
                elif free_param == "E_constant":
                    E_constant = tf.expand_dims(tf.expand_dims(tf.expand_dims(z[:, :, ind], 1), 3), 4)
                elif free_param == "E_Pbias":
                    E_Pbias = tf.expand_dims(tf.expand_dims(tf.expand_dims(z[:, :, ind], 1), 3), 4)
                elif free_param == "E_Prule":
                    E_Prule = tf.expand_dims(tf.expand_dims(tf.expand_dims(z[:, :, ind], 1), 3), 4)
                elif free_param == "E_Arule":
                    E_Arule = tf.expand_dims(tf.expand_dims(tf.expand_dims(z[:, :, ind], 1), 3), 4)
                elif free_param == "E_choice":
                    E_choice = tf.expand_dims(tf.expand_dims(tf.expand_dims(z[:, :, ind], 1), 3), 4)
                elif free_param == "E_light":
                    E_light = tf.expand_dims(tf.expand_dims(tf.expand_dims(z[:, :, ind], 1), 3), 4)

                else:
                    print("Error: unknown free parameter: %s." % free_param)
                    raise NotImplementedError()
                ind += 1

            # load fixed parameters
            for fixed_param in self.fixed_params.keys():
                if fixed_param == "sW":
                    sW = self.fixed_params[fixed_param] * tf.ones(
                        (self.C, M), dtype=DTYPE
                    )
                elif fixed_param == "vW":
                    vW = self.fixed_params[fixed_param] * tf.ones(
                        (self.C, M), dtype=DTYPE
                    )
                elif fixed_param == "dW":
                    dW = self.fixed_params[fixed_param] * tf.ones(
                        (self.C, M), dtype=DTYPE
                    )
                elif fixed_param == "hW":
                    hW = self.fixed_params[fixed_param] * tf.ones(
                        (self.C, M), dtype=DTYPE
                    )
                elif fixed_param == "E_constant":
                    E_constant = self.fixed_params[fixed_param] * tf.ones((1, 1, M, 1, 1), dtype=DTYPE)
                elif fixed_param == "E_Pbias":
                    E_Pbias = self.fixed_params[fixed_param] * tf.ones((1, 1, M, 1, 1), dtype=DTYPE)
                elif fixed_param == "E_Prule":
                    E_Prule = self.fixed_params[fixed_param] * tf.ones((1, 1, M, 1, 1), dtype=DTYPE)
                elif fixed_param == "E_Arule":
                    E_Arule = self.fixed_params[fixed_param] * tf.ones((1, 1, M, 1, 1), dtype=DTYPE)
                elif fixed_param == "E_choice":
                    E_choice = self.fixed_params[fixed_param] * tf.ones((1, 1, M, 1, 1), dtype=DTYPE)
                elif fixed_param == "E_light":
                    E_light = self.fixed_params[fixed_param] * tf.ones((1, 1, M, 1, 1), dtype=DTYPE)

                else:
                    print("Error: unknown fixed parameter: %s." % fixed_param)
                    raise NotImplementedError()

            # Gather weights into the dynamics matrix W [C,M,4,4]
            Wrow1 = tf.stack([sW, vW, dW, hW], axis=2)
            Wrow2 = tf.stack([vW, sW, hW, dW], axis=2)
            Wrow3 = tf.stack([dW, hW, sW, vW], axis=2)
            Wrow4 = tf.stack([hW, dW, vW, sW], axis=2)
            W = tf.stack([Wrow1, Wrow2, Wrow3, Wrow4], axis=2)

        # input current time courses
        I_constant = E_constant*tf.ones((self.T, 1, 1, 4, 1), dtype=DTYPE)

        I_Pbias = np.zeros((self.T,4))
        I_Pbias[self.t < 1.2] = np.array([1, 0, 0, 1])
        I_Pbias = np.expand_dims(np.expand_dims(np.expand_dims(I_Pbias, 2), 1), 1)
        I_Pbias = E_Pbias*tf.constant(I_Pbias)

        I_Prule = np.zeros((self.T,4))
        I_Prule[self.t < 1.2] = np.array([1, 0, 0, 1])
        I_Prule = np.expand_dims(np.expand_dims(np.expand_dims(I_Prule, 2), 1), 1)
        I_Prule = E_Prule*tf.constant(I_Prule)

        I_Arule = np.zeros((self.T,4))
        I_Arule[self.t < 1.2] = np.array([0, 1, 1, 0])
        I_Arule = np.expand_dims(np.expand_dims(np.expand_dims(I_Arule, 2), 1), 1)
        I_Arule = E_Arule*tf.constant(I_Arule)

        I_choice = np.zeros((self.T,4))
        I_choice[self.t > 1.2] = np.array([1, 1, 1, 1])
        I_choice = np.expand_dims(np.expand_dims(np.expand_dims(I_choice, 2), 1), 1)
        I_choice = E_choice*tf.constant(I_choice)

        I_lightL = np.zeros((self.T,4))
        I_lightL[self.t > 1.2] = np.array([1, 1, 0, 0])
        I_lightL = np.expand_dims(np.expand_dims(np.expand_dims(I_lightL, 2), 1), 1)
        I_lightL = E_light*tf.constant(I_lightL)

        I_lightR = np.zeros((self.T,4))
        I_lightR[self.t > 1.2] = np.array([0, 0, 1, 1])
        I_lightR = np.expand_dims(np.expand_dims(np.expand_dims(I_lightR, 2), 1), 1)
        I_lightR = E_light*tf.constant(I_lightR)

        # Gather inputs into I [T,C,1,4,1]
        if self.behavior["type"] in ["standard", "means", "pvar", "feasible"]:
            I_LP = I_constant + I_Pbias + I_Prule + I_choice + I_lightL
            I = I_LP

        
        # construct input I

        return W, I


    def simulate(self, z):
        """Simulate the V1 4-neuron circuit given parameters z.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            g(z) (tf.tensor): Simulated system activity.

        """

        # get number of batch samples
        z_shape = tf.shape(z)
        K = z_shape[0]
        M = z_shape[1]

        # set constant parameters
        theta = 0.05
        beta = 0.5
        tau = 0.09
        sigma = 0.3


        # obtain weights and inputs from parameterization
        W, I = self.filter_Z(z)

        # initial conditions
        v0 = 0.1*tf.ones((self.C, M, 4, self.N), dtype=DTYPE)
        u0 = beta*tf.math.atanh(2*v0-1) - theta

        v = v0
        u = u0
        v_t_list = [v]
        u_t_list = [u]
        for i in range(1,self.T):
            du = (self.dt /tau) * (-u + tf.matmul(W, v) + I[i] + sigma*self.w[i])
            u = u + du
            v = 0.5*tf.tanh((u - theta)/beta) + 0.5
            v_t_list.append(v)
            u_t_list.append(u)

        v_t = tf.stack(v_t_list, axis=0)
        return v_t

    def compute_suff_stats(self, z):
        """Compute sufficient statistics of density network samples.

        Behaviors:

        'standard' - 

          Add a description.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            T_x (tf.tensor): Sufficient statistics of samples.

        """

        if self.behavior["type"] in ["standard", "means", "pvar", "feasible"]:
            T_x = self.simulation_suff_stats(z)
        else:
            raise NotImplementedError()

        return T_x

    def simulation_suff_stats(self, z):
        """Compute sufficient statistics that require simulation.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            T_x (tf.tensor): Simulation-derived sufficient statistics of samples.

        """

        v_t = self.simulate(z)
        # [T, C, M, D, 1]

        
        v_LP = v_t[-1, 0, :, 0, :] # we're looking at LP in the standard L Pro condition
        E_v_LP = tf.reduce_mean(v_LP, 1)
        Var_v_LP = tf.reduce_mean(tf.square(v_LP - tf.expand_dims(E_v_LP, 1)), 1)

        # Add leading dimension
        E_v_LP = tf.expand_dims(E_v_LP, 0)
        Var_v_LP = tf.expand_dims(Var_v_LP, 0)
        Bern_Var_Err = Var_v_LP - (E_v_LP*(1-E_v_LP))

        if self.behavior["type"] == "standard":
            T_x = tf.stack((E_v_LP, \
                            Bern_Var_Err, \
                            tf.square(E_v_LP), \
                            tf.square(Bern_Var_Err)
                            ), 2)
        elif self.behavior["type"] == "means":
            T_x = tf.stack((E_v_LP, \
                            Bern_Var_Err
                            ), 2)
        elif self.behavior["type"] == "pvar":
            T_x = tf.stack((E_v_LP, \
                            Bern_Var_Err, \
                            tf.square(E_v_LP)
                            ), 2)
        if self.behavior["type"] == "feasible":
            T_x = tf.stack((Var_v_LP, \
                            tf.square(Var_v_LP)
                            ), 2)
        else:
            raise NotImplementedError()

        return T_x

    def compute_mu(self,):
        """Calculate expected moment constraints given system paramterization.

        # Returns
            mu (np.array): Expected moment constraints.

        """

        means = self.behavior["means"]
        first_moments = means
        if self.behavior["type"] in ["standard", "feasible"]:
            variances = self.behavior["variances"]
            second_moments = np.square(means) + variances
            mu = np.concatenate((first_moments, second_moments), axis=0)
        elif self.behavior["type"] == "means":
            mu = first_moments
        elif self.behavior["type"] == "pvar":
            pvar = self.behavior["pvar"]
            p_hat_second_moment = np.square(np.array([means[0]])) + pvar
            mu = np.concatenate((first_moments, p_hat_second_moment), axis=0)
        return mu



class LowRankRNN(system):
    """ Recent work by ([Mastrogiusseppe & Ostojic, 2018](#Mastrogiuseppe2018Linking)) allows us to 
        derive statistical properties of the behavior of recurrent 
        neural networks (RNNs) given a low-rank parameterization of 
        their connectivity.  This work builds on dynamic mean field 
        theory (DMFT) for neural networks (Sompolinsky et al. 1988), 
        which is exact in the limit of infinite neurons, but has been 
        shown to yield accurate approximations for finite size 
        networks.

        The network model is

        $$\dot{x}_i(t) = -x_i(t) + \sum_{j=1}^N J_{ij} \phi(x_j(t)) + I_i $$

        where the connectivity is comprised of a random and structured component:

        $$J_{ij} = g \chi_{ij} + P_{ij}$$
        
        The random all-to-all component has elements drawn from 
        $$\chi_{ij} \sim \mathcal{N}(0, \\frac{1}{N})$$, and the structured
        component is a sum of $$r$$ unit rank terms:

        $$P_{ij} = \sum_{k=1}^r \\frac{m_i^{(k)}n_j^{(k)}}{N}$$

        The nonlinearity $$\phi$$ is set to $$tanh$$ in this software, but
        the theory is general for many other activation functions.


    # Attributes
        behavior (dict): see LowRankRNN.compute_suff_stats
        model_opts (dict): 
          * model_opts[`'rank'`] 
            * `1` (default) Rank 1 network
            * `2` 
          * model_opts[`'input_type'`] 
            * `'spont'` (default) No input.
            * `'gaussian'` (default) Gaussian input.
        solve_its (int): Number of langevin dynamics simulation steps.
        solve_eps (float): Langevin dynamics solver step-size.
    """

    def __init__(
        self,
        fixed_params,
        behavior,
        model_opts={"rank": 1, "input_type": "spont"},
        solve_its=25,
        solve_eps=0.8,
    ):
        self.model_opts = model_opts
        super().__init__(fixed_params, behavior)
        self.name = "LowRankRNN"
        self.solve_its = solve_its
        self.solve_eps = solve_eps

    def get_all_sys_params(self,):
        """Returns ordered list of all system parameters and individual element labels.

        When `model_opts['rank'] == 1`

         - $$g$$ - strength of the random matrix component
         - $$M_m$$ - mean value of right connectivity vector
         - $$M_n$$ - mean value of left connectivity vector
         - $$\Sigma_m$$ - variance of values in right connectivity vector

        When `model_opts['rank'] == 2`
         
         - TODO

        # Returns
            all_params (list): List of strings of all parameters of full system model.
            all_param_labels (list): List of tex strings for all parameters.
        """

        if (self.model_opts['rank'] == 1 and self.model_opts['input_type'] == 'spont'):
            all_params = ["g", "Mm", "Mn", "Sm"]
            all_param_labels = {
                "g": [r"$g$"],
                "Mm": [r"$M_m$"],
                "Mn": [r"$M_n$"],
                "Sm": [r"$\Sigma_m$"],
            }
        elif (self.model_opts['rank'] == 1 and self.model_opts['input_type'] == 'input'):
            all_params = ["g", "Mm", "Mn", "MI", "Sm", "Sn", "SmI", "Sperp"]
            all_param_labels = {
                "g": [r"$g$"],
                "Mm": [r"$M_m$"],
                "Mn": [r"$M_n$"],
                "MI": [r"$M_I$"],
                "Sm": [r"$\Sigma_m$"],
                "Sn": [r"$\Sigma_n$"],
                "SmI": [r"$\Sigma_{m,I}$"],
                "Sperp": [r"$\Sigma_\perp$"],
            }
        elif (self.model_opts['rank'] == 2 and self.model_opts['input_type'] == 'input' and self.behavior['type'] == 'CDD'):
            all_params = ["g", "rhom", "rhon", "betam", "betan", "gammaLO", "gammaHI"]
            all_param_labels = {
                "g": [r"$g$"],
                "rhom": [r"$\rho_m$"],
                "rhon": [r"$\rho_n$"],
                "betam": [r"$\beta_m$"],
                "betan": [r"$\beta_n$"],
                "gammaLO": [r"$\gamma_{LO}$"],
                "gammaHI": [r"$\gamma_{HI}$"],
            }
        return all_params, all_param_labels

    def get_T_x_labels(self,):
        """Returns `T_x_labels`.

        Behaviors:

        'struct_chaos' - $$[\mu, \Delta_{\infty}, (\Delta_0 - \Delta_{\infty}), \mu^2, \Delta_{\infty}^2, (\Delta_0 - \Delta_{\infty})^2]$$
        
        # Returns
            T_x_labels (list): List of tex strings for elements of $$T(x)$$.

        """
        if self.behavior["type"] == "struct_chaos":
            T_x_labels = [
                r"$\mu$",
                r"$\Delta_{\infty}$",
                r"$\Delta_T$",
                r"$\mu^2$",
                r"$\Delta_{\infty}^2$",
                r"$(\Delta_T)^2$",
            ]
        elif self.behavior["type"] == "ND":
            T_x_labels = [
                r"$\kappa_{HI}  -\kappa_{LO}$",
                #r"$\Delta_T$",
                r"$(\kappa_{HI}  -\kappa_{LO})^2$",
                #r"$\Delta_T^2$",
            ]
        elif self.behavior["type"] == "CDD":
            T_x_labels = [
                r"$z_{ctxA,A}$",
                r"$z_{ctxA,B}$",
                r"$z_{ctxA,A}^2$",
                r"$z_{ctxA,B}^2$",
            ]
        else:
            raise NotImplementedError()
        return T_x_labels

    def filter_Z(self, z):
        """Returns the system matrix/vector variables depending free parameter ordering.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            g (tf.tensor): [1,M] Strength of the random matrix component.
            Mm (tf.tensor): [1,M] Mean value of right connectivity vector.
            Mn (tf.tensor): [1,M] Mean value of left connectivity vector.
            Sm (tf.tensor): [1,M] Variance of values in right connectivity vector.

        """

        z_shape = tf.shape(z)
        K = z_shape[0]
        M = z_shape[1]

        # read free parameters from z vector
        ind = 0

        if (self.model_opts['rank'] == 1 and self.model_opts['input_type'] == 'spont'):
            for free_param in self.free_params:
                if free_param == "g":
                    g = z[:, :, ind]
                elif free_param == "Mm":
                    Mm = z[:, :, ind]
                elif free_param == "Mn":
                    Mn = z[:, :, ind]
                elif free_param == "Sm":
                    Sm = z[:, :, ind]
                else:
                    print("Error: unknown free parameter: %s." % free_param)
                    raise NotImplementedError()
                ind += 1

            # load fixed parameters
            for fixed_param in self.fixed_params.keys():
                if fixed_param == "g":
                    g = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
                elif fixed_param == "Mm":
                    Mm = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
                elif fixed_param == "Mn":
                    Mn = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
                elif fixed_param == "Sm":
                    Sm = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
                else:
                    print("Error: unknown fixed parameter: %s." % fixed_param)
                    raise NotImplementedError()

            return g, Mm, Mn, Sm

        elif (self.model_opts['rank'] == 1 and self.model_opts['input_type'] == 'input'):
            for free_param in self.free_params:
                if free_param == "g":
                    g = z[:, :, ind]
                elif free_param == "Mm":
                    Mm = z[:, :, ind]
                elif free_param == "Mn":
                    Mn = z[:, :, ind]
                elif free_param == "MI":
                    MI = z[:, :, ind]
                elif free_param == "Sm":
                    Sm = z[:, :, ind]
                elif free_param == "Sn":
                    Sn = z[:, :, ind]
                elif free_param == "SmI":
                    SmI = z[:, :, ind]
                elif free_param == "Sperp":
                    Sperp = z[:, :, ind]
                else:
                    print("Error: unknown free parameter: %s." % free_param)
                    raise NotImplementedError()
                ind += 1

            # load fixed parameters
            for fixed_param in self.fixed_params.keys():
                if fixed_param == "g":
                    g = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
                elif fixed_param == "Mm":
                    Mm = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
                elif fixed_param == "Mn":
                    Mn = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
                elif fixed_param == "MI":
                    MI = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
                elif fixed_param == "Sm":
                    Sm = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
                elif fixed_param == "Sn":
                    Sn = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
                elif fixed_param == "SmI":
                    SmI = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
                elif fixed_param == "Sperp":
                    Sperp = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
                else:
                    print("Error: unknown fixed parameter: %s." % fixed_param)
                    raise NotImplementedError()

            return g, Mm, Mn, MI, Sm, Sn, SmI, Sperp

        elif (self.model_opts['rank'] == 2 and self.model_opts['input_type'] == 'input' and self.behavior['type'] == 'CDD'):
            for free_param in self.free_params:
                if free_param == "g":
                    g = z[:, :, ind]
                elif free_param == "rhom":
                    rhom = z[:, :, ind]
                elif free_param == "rhon":
                    rhon = z[:, :, ind]
                elif free_param == "betam":
                    betam = z[:, :, ind]
                elif free_param == "betan":
                    betan = z[:, :, ind]
                elif free_param == "gammaLO": # negate
                    gammaLO = -z[:, :, ind]
                elif free_param == "gammaHI":
                    gammaHI = z[:, :, ind]
                else:
                    print("Error: unknown free parameter: %s." % free_param)
                    raise NotImplementedError()
                ind += 1

            # load fixed parameters
            for fixed_param in self.fixed_params.keys():
                if fixed_param == "g":
                    g = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
                elif fixed_param == "rhom":
                    rhom = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
                elif fixed_param == "rhon":
                    rhon = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
                elif fixed_param == "betam":
                    betam = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
                elif fixed_param == "betan":
                    betan = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
                elif fixed_param == "gammaLO":
                    gammaLO = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
                elif fixed_param == "gammaHI":
                    gammaHI = self.fixed_params[fixed_param] * tf.ones((1, M), dtype=DTYPE)
                else:
                    print("Error: unknown fixed parameter: %s." % fixed_param)
                    raise NotImplementedError()

            return g, rhom, rhon, betam, betan, gammaLO, gammaHI

    def compute_suff_stats(self, z):
        """Compute sufficient statistics of density network samples.

        Behaviors:

        'struct_chaos' - 

          When `model_opts['rank'] == 1` and `model_opts['input_type'] == 'spont'`

          Set constraints on the mean unit activity $$\mu$$, the static variance
          $$\Delta_{\infty}$$, and the temporal variance $$\Delta_T = \Delta_0 - \Delta_{\infty}$$.
          $$\mu$$, $$\Delta_0$$, and $$\Delta_{\infty}$$ can be found for a rank-1
          no-input network by solving the following consistency equations.

          $$\mu = F(\mu, \Delta_0, \Delta_\infty) = M_m M_n \int \mathcal{D}z \phi(\mu + \sqrt{\Delta_0} z)$$

          $$\Delta_0 = G(\mu, \Delta_0, \Delta_\infty) = [\Delta_\infty^2 + 2g^2\{\int \mathcal{D}z \Phi^2(\mu + \sqrt{\Delta_0}z) - \int \mathcal{D}z [\int \mathcal{D}x \Phi(\mu + \sqrt{\Delta_0 - \Delta_\infty}x $$
          $$ + \sqrt{\Delta_\infty}z)]^2\} +M_n^2 \Sigma_m^2 \langle[\phi_i]\\rangle^2(\Delta_0 - \Delta_\infty)]^{\\frac{1}{2}} $$

          $$\Delta_\infty = H(\mu, \Delta_0, \Delta_\infty) = g^2 \int \mathcal{D}z \left[ \int \mathcal{D}x \Phi(\mu + \sqrt{\Delta_0 - \Delta_\infty} + \sqrt{\Delta_\infty}z \\right]^2 + M_n^2 \Sigma_m^2 \langle [\phi_i] \\rangle^2$$

          The solutions are found via a Langevin dynamics simulation with step size 
          `self.solve_eps` and number of iterations `self.solve_its`.

          $$\dot{\mu} = -\mu + F(\mu, \Delta_0, \Delta_\infty)$$

          $$\dot{\Delta_0} = -\Delta_0 + G(\mu, \Delta_0, \Delta_\infty)$$

          $$\dot{\Delta_\infty} = -\Delta_\infty + H(\mu, \Delta_0, \Delta_\infty)$$
        
          The total constraint vector is
          \\begin{equation}
          E_{x\\sim p(x \\mid z)}\\left[T(x)\\right] = \\begin{bmatrix} \mu \\\\\\\\ \Delta_\infty \\\\\\\\ \Delta_0 - \Delta_\infty \\\\\\\\ \mu \\\\\\\\ \Delta_\infty^2 \\\\\\\\ (\Delta_0 - \Delta_\infty)^2 \end{bmatrix}
          \end{equation}

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            T_x (tf.tensor): Sufficient statistics of samples.

        """

        M = tf.shape(z)[1]

        if self.behavior["type"] == "struct_chaos":
            if self.model_opts["input_type"] == "spont":
                g, Mm, Mn, Sm = self.filter_Z(z)
                mu_init = 50.0 * tf.ones((M,), dtype=DTYPE)
                delta_0_init = 55.0 * tf.ones((M,), dtype=DTYPE)
                delta_inf_init = 45.0 * tf.ones((M,), dtype=DTYPE)

                mu, delta_0, delta_inf = rank1_spont_chaotic_solve(
                    mu_init,
                    delta_0_init,
                    delta_inf_init,
                    g[0, :],
                    Mm[0, :],
                    Mn[0, :],
                    Sm[0, :],
                    self.solve_its,
                    self.solve_eps,
                    gauss_quad_pts=50,
                    db=False
                )

                static_var = delta_inf
                chaotic_var = delta_0 - delta_inf

                first_moments = tf.stack([mu, static_var, chaotic_var], axis=1)
                second_moments = tf.square(first_moments)
                T_x = tf.expand_dims(
                    tf.concat((first_moments, second_moments), axis=1), 0
                )

            else:
                raise NotImplementedError()

        elif self.behavior["type"] == "ND":
            assert(self.model_opts["input_type"] == "input")
            num_conds = 2
            c_LO = 0.25
            c_HI = 0.75

            g, Mm, Mn, MI, Sm, Sn, SmI, Sperp = self.filter_Z(z)
            g, Mm, Mn, MI, Sm, Sn, SmI, Sperp = tile_for_conditions(
                [g, Mm, Mn, MI, Sm, Sn, SmI, Sperp], 
                num_conds)

            mu_init = -5.0 * tf.ones((num_conds*M,), dtype=DTYPE)
            kappa_init = -5.0 * tf.ones((num_conds*M,), dtype=DTYPE)
            delta_0_init = 5.0 * tf.ones((num_conds*M,), dtype=DTYPE)
            delta_inf_init = 4.0 * tf.ones((num_conds*M,), dtype=DTYPE)

            SnI = tf.concat((c_LO*tf.ones((M,), dtype=DTYPE), 
                             c_HI*tf.ones((M,), dtype=DTYPE)), 
                            axis=0)
            
            mu, kappa, delta_0, delta_inf = rank1_input_chaotic_solve(
                mu_init,
                kappa_init,
                delta_0_init,
                delta_inf_init,
                g[0, :],
                Mm[0, :],
                Mn[0, :],
                MI[0, :],
                Sm[0, :],
                Sn[0, :],
                SmI[0, :],
                SnI,
                Sperp[0, :],
                self.solve_its,
                self.solve_eps,
                gauss_quad_pts=50,
                db=False)

            #static_var = delta_inf
            kappa_LO = kappa[:M]
            kappa_HI = kappa[M:]

            first_moments = tf.stack([kappa_HI-kappa_LO], axis=1)
            second_moments = tf.square(first_moments)
            T_x = tf.expand_dims(
                tf.concat((first_moments, second_moments), axis=1), 0
            )

                
        elif self.behavior["type"] == "CDD":
            num_conds = 2
            c_LO = 0.0
            c_HI = 1.0

            g, rhom, rhon, betam, betan, gammaLO, gammaHI = self.filter_Z(z)
            gammaHI, gammaLO = tile_for_conditions([gammaHI, gammaLO], 2)

            g, rhom, rhon, betam, betan = tile_for_conditions(
                [g, rhom, rhon, betam, betan], 
                num_conds)


            gammaA = gammaHI
            gammaB = gammaLO

            cA = tf.concat((c_HI*tf.ones((M,), dtype=DTYPE), 
                             c_LO*tf.ones((M,), dtype=DTYPE)),
                             axis=0)
            cB = tf.concat((c_LO*tf.ones((M,), dtype=DTYPE), 
                             c_HI*tf.ones((M,), dtype=DTYPE)),
                             axis=0)

            


            kappa1_init = -5.0 * tf.ones((num_conds*M,), dtype=DTYPE)
            kappa2_init = -5.0 * tf.ones((num_conds*M,), dtype=DTYPE)
            delta_0_init = 5.0 * tf.ones((num_conds*M,), dtype=DTYPE)
            #delta_inf_init = 4.0 * tf.ones((num_conds*M,), dtype=DTYPE)

            # TODO delta_0 should be written square diff in commented out?
            #kappa1, kappa2, delta_0, delta_inf, z = rank2_CDD_chaotic_solve(
            kappa1, kappa2, delta_0, z = rank2_CDD_static_solve(
                kappa1_init,
                kappa2_init,
                delta_0_init,
                cA,
                cB,
                g[0, :],
                rhom[0, :],
                rhon[0, :],
                betam[0, :],
                betan[0, :],
                gammaA[0,:],
                gammaB[0,:],
                self.solve_its,
                self.solve_eps,
                gauss_quad_pts=50,
                db=False,
            )

            z_ctxA_A = z[:M]
            z_ctxA_B = z[M:2*M]

            first_moments = tf.stack([z_ctxA_A, z_ctxA_B], axis=1)
            second_moments = tf.square(first_moments)
            T_x = tf.expand_dims(
                tf.concat((first_moments, second_moments), axis=1), 0
            )
        
        else:
            raise NotImplementedError()

        return T_x

    def compute_mu(self,):
        """Calculate expected moment constraints given system paramterization.

        # Returns
            mu (np.array): Expected moment constraints.

        """

        if self.behavior["type"] in ["struct_chaos", "ND", "CDD"]:
            means = self.behavior["means"]
            variances = self.behavior["variances"]
        else:
            raise NotImplementedError()
        first_moments = means
        second_moments = np.square(means) + variances
        mu = np.concatenate((first_moments, second_moments), axis=0)
        return mu

    def support_mapping(self, inputs):
        """Maps from real numbers to support of parameters.

        # Arguments:
            inputs (np.array): Input from previous layers of the DSN.

        # Returns
            Z (np.array): Samples from the DSN at the final layer.
        """
        return SoftPlusFlow([], inputs)


def system_from_str(system_str):
    if system_str in ["Linear2D"]:
        return Linear2D
    elif system_str in ["damped_harmonic_oscillator", "dho"]:
        return damped_harmonic_oscillator
    elif system_str in ["rank1_rnn"]:
        return RNN_rank1
    elif system_str in ["R1RNN_input"]:
        return R1RNN_input
    elif system_str in ["R1RNN_GNG"]:
        return R1RNN_GNG
    elif system_str in ["V1Circuit"]:
        return V1Circuit
    elif system_str in ["SCCircuit"]:
        return SCCircuit
