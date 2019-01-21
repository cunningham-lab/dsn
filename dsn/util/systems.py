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
from dsn.util import tf_integrals as tf_integrals


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
            if (not param_str in self.fixed_params.keys()):
                free_params.append(param_str)
        return free_params

    def get_z_labels(self,):
        """Returns `z_labels`.

        # Returns
            z_labels (list): List of tex strings for free parameters.

        """
        z_labels = [];
        for free_param in self.free_params:
            z_labels += self.all_param_labels[free_param]
        return z_labels

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
        all_params = ['A', 'tau'];
        all_param_labels = {'A':[r'$a_1$', r'$a_2$', r'$a_3$', r'$a_4$'], \
                            'tau':[r'$\tau$']}
        return all_params, all_param_labels

    def get_T_x_labels(self,):
        """Returns `T_x_labels`.

        Behaviors:

        'oscillation' - $$[$$real($$\lambda_1$$), $$\\frac{\\text{imag}(\lambda_1)}{2 \pi}$$, real$$(\lambda_1)^2$$, $$(\\frac{\\text{imag}(\lambda_1)}{2 \pi})^2]$$

        # Returns
            T_x_labels (list): List of tex strings for elements of $$T(x)$$.

        """
        if (self.behavior['type']== 'oscillation'):
            T_x_labels = [r'real($\lambda_1$)', r'imag($\lambda_1$)', \
                          r'real$(\lambda_1)^2$', r'imag$(\lambda_1)^2$'];
        else:
            raise NotImplementedError()
        return T_x_labels;


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
        if self.behavior['type'] == "oscillation":
            z_shape = tf.shape(z)
            K = z_shape[0]
            M = z_shape[1]

            # read free parameters from z vector
            ind = 0;
            for free_param in self.free_params:
                if (free_param == 'A'):
                    a1 = z[:, :, ind]
                    a2 = z[:, :, ind+1]
                    a3 = z[:, :, ind+2]
                    a4 = z[:, :, ind+3]
                    ind += 4
                elif (free_param == 'tau'):
                    tau = z[:, :, ind]
                    ind += 1

            # load fixed parameters
            for fixed_param in self.fixed_params.keys():
                if (fixed_param == 'A'):
                    a1 = self.fixed_params['A'][0]
                    a2 = self.fixed_params['A'][1]
                    a3 = self.fixed_params['A'][2]
                    a4 = self.fixed_params['A'][3]
                elif (fixed_param == 'tau'):
                    tau = self.fixed_params['tau']

            # C = A / tau are the effective linear dynamics 
            c1 = tf.divide(a1, tau)
            c2 = tf.divide(a2, tau)
            c3 = tf.divide(a3, tau)
            c4 = tf.divide(a4, tau)

            beta = tf.complex(tf.square(c1 + c4) - 4 * (c1 * c4 - c2 * c3), np.float64(0.0))
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
        types.  Additionally, much work has been done to measure the relative magnitudes
        of the synaptic projections between neural types.
        \\begin{equation}
        W = \\begin{bmatrix} W_{EE} & -1.0 & -0.54 & 0 \\\\\\\\ W_{PE} & -1.01 & -0.33 & 0 \\\\\\\\ W_{SE} & 0 & 0 & -0.15 \\\\\\\\ W_{VE} & -0.22 & -0.77 & 0 \end{bmatrix}
        \end{equation}

        In this model, we are interested in capturing V1 responses across varying 
        contrasts $$c$$, stimulus sizes $$s$$, and locomotion $$r$$ conditions.

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

    def __init__(self, fixed_params, behavior, \
                 model_opts={'g_FF':'c', 'g_LAT':'linear', 'g_RUN':'r'}, \
                 T=40, dt=0.25, \
                 init_conds=np.expand_dims(np.array([1.0, 1.1, 1.2, 1.3]), 1)):
        self.model_opts = model_opts
        super().__init__(fixed_params, behavior)
        self.name = "V1Circuit"
        self.T = T
        self.dt = dt
        self.init_conds = init_conds

        # compute number of conditions C
        num_c = self.behavior['c_vals'].shape[0]
        num_s = self.behavior['s_vals'].shape[0]
        num_r = self.behavior['r_vals'].shape[0]
        self.C = num_c*num_s*num_r


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
        all_params = ['W_EE', 'W_PE', 'W_SE', 'W_VE', \
                      'b_E', 'b_P', 'b_S', 'b_V', \
                      'h_FFE', 'h_FFP', \
                      'h_LATE', 'h_LATP', 'h_LATS', 'h_LATV', \
                      'h_RUNE', 'h_RUNP', 'h_RUNS', 'h_RUNV', \
                      'tau', 'n', 's_0']
        all_param_labels = {'W_EE':[r'$W_{EE}$'], 'W_PE':[r'$W_{PE}$'], 'W_SE':[r'$W_{SE}$'], 'W_VE':[r'$W_{VE}$'], \
                      'b_E':[r'$b_{E}$'], 'b_P':[r'$b_{P}$'], 'b_S':[r'$b_{S}$'], 'b_V':[r'$b_{V}$'], \
                      'h_FFE':[r'$h_{FF,E}$'], 'h_FFP':[r'$h_{FF,P}$'], \
                      'h_LATE':[r'$h_{LAT,E}$'], 'h_LATP':[r'$h_{LAT,P}$'], 'h_LATS':[r'$h_{LAT,S}$'], 'h_LATV':[r'$h_{LAT,V}$'], \
                      'h_RUNE':[r'$h_{RUN,E}$'], 'h_RUNP':[r'$h_{RUN,P}$'], 'h_RUNS':[r'$h_{RUN,S}$'], 'h_RUNV':[r'$h_{RUN,V}$'], \
                      'tau':[r'$\tau$'], 'n':[r'$n$'], 's_0':[r'$s_0$']}

        if (self.model_opts['g_FF'] == 'saturate'):
            all_params +=  ['a', 'c_50']
            all_param_labels.update({'a':r'$a$', 'c_50':r'$c_{50}$'})

        return all_params, all_param_labels

    def get_T_x_labels(self,):
        """Returns `T_x_labels`.

        Behaviors:

        'difference' - $$[d_{E,ss}, d_{P,ss}, d_{S,ss}, d_{V,ss}, d_{E,ss}^2, d_{P,ss}^2, d_{S,ss}^2, d_{V,ss}^2]$$
        
        'data' - $$[r_{E,ss}(c,s,r), ...,  r_{E,ss}(c,s,r)^2, ...]$$

        # Returns
            T_x_labels (list): List of tex strings for elements of $$T(x)$$.

        """
        if (self.behavior['type'] == 'difference'):
            all_T_x_labels = [r'$d_{E,ss}$', r'$d_{P,ss}$', r'$d_{S,ss}$', r'$d_{V,ss}$', \
                              r'$d_{E,ss}^2$', r'$d_{P,ss}^2$', r'$d_{S,ss}^2$', r'$d_{V,ss}^2$']
            diff_inds = self.behavior['diff_inds']
            label_inds = diff_inds + list(map(lambda x : x + 4, diff_inds))
            T_x_labels = []
            for i in range(len(label_inds)):
                T_x_labels.append(all_T_x_labels[label_inds[i]])
        else:
            raise NotImplementedError()
        return T_x_labels;

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
        W_EP = 1.0*tf.ones((self.C,M), dtype=tf.float64)
        W_ES = 0.54*tf.ones((self.C,M), dtype=tf.float64)

        W_PP = 1.01*tf.ones((self.C,M), dtype=tf.float64)
        W_PS = 0.33*tf.ones((self.C,M), dtype=tf.float64)

        W_SV = 0.15*tf.ones((self.C,M), dtype=tf.float64)

        W_VP = 0.22*tf.ones((self.C,M), dtype=tf.float64)
        W_VS = 0.77*tf.ones((self.C,M), dtype=tf.float64)

        # read free parameters from z vector
        ind = 0;
        for free_param in self.free_params:
            if (free_param == 'W_EE'):
                W_EE = tf.tile(z[:, :, ind], [self.C, 1])
            elif (free_param == 'W_PE'):
                W_PE = tf.tile(z[:, :, ind], [self.C, 1])
            elif (free_param == 'W_SE'):
                W_SE = tf.tile(z[:, :, ind], [self.C, 1])
            elif (free_param == 'W_VE'):
                W_VE = tf.tile(z[:, :, ind], [self.C, 1])

            elif (free_param == 'b_E'):
                b_E = tf.tile(z[:, :, ind], [1, 1])
            elif (free_param == 'b_P'):
                b_P = tf.tile(z[:, :, ind], [1, 1])
            elif (free_param == 'b_S'):
                b_S = tf.tile(z[:, :, ind], [1, 1])
            elif (free_param == 'b_V'):
                b_V = tf.tile(z[:, :, ind], [1, 1])

            elif (free_param == 'h_FFE'):
                h_FFE = tf.tile(z[:, :, ind], [1, 1])
            elif (free_param == 'h_FFP'):
                h_FFP = tf.tile(z[:, :, ind], [1, 1])

            elif (free_param == 'h_LATE'):
                h_LATE = tf.tile(z[:, :, ind], [1, 1])
            elif (free_param == 'h_LATP'):
                h_LATP = tf.tile(z[:, :, ind], [1, 1])
            elif (free_param == 'h_LATS'):
                h_LATS = tf.tile(z[:, :, ind], [1, 1])
            elif (free_param == 'h_LATV'):
                h_LATV = tf.tile(z[:, :, ind], [1, 1])

            elif (free_param == 'h_RUNE'):
                h_RUNE = tf.tile(z[:, :, ind], [1, 1])
            elif (free_param == 'h_RUNP'):
                h_RUNP = tf.tile(z[:, :, ind], [1, 1])
            elif (free_param == 'h_RUNS'):
                h_RUNS = tf.tile(z[:, :, ind], [1, 1])
            elif (free_param == 'h_RUNV'):
                h_RUNV = tf.tile(z[:, :, ind], [1, 1])

            elif (free_param == 'tau'):
                tau = tf.tile(z[:, :, ind], [self.C, 1])
            elif (free_param == 'n'):
                n = tf.tile(z[:, :, ind], [self.C, 1])
            elif (free_param == 's_0'):
                s_0 = tf.tile(z[:, :, ind], [1, 1])
            elif (free_param == 'a'):
                a = tf.tile(z[:, :, ind], [1, 1])
            elif (free_param == 'c_50'):
                c_50 = tf.tile(z[:, :, ind], [1, 1])

            else:
                print('Error: unknown free parameter: %s.' % free_param)
                raise NotImplementedError()
            ind += 1

        # load fixed parameters
        for fixed_param in self.fixed_params.keys():
            if (fixed_param == 'W_EE'):
                W_EE = self.fixed_params[fixed_param]*tf.ones((self.C,M), dtype=tf.float64);
            elif (fixed_param == 'W_PE'):
                W_PE = self.fixed_params[fixed_param]*tf.ones((self.C,M), dtype=tf.float64);
            elif (fixed_param == 'W_SE'):
                W_SE = self.fixed_params[fixed_param]*tf.ones((self.C,M), dtype=tf.float64);
            elif (fixed_param == 'W_VE'):
                W_VE = self.fixed_params[fixed_param]*tf.ones((self.C,M), dtype=tf.float64);

            elif (fixed_param == 'b_E'):
                b_E = self.fixed_params[fixed_param]*tf.ones((1,M), dtype=tf.float64);
            elif (fixed_param == 'b_P'):
                b_P = self.fixed_params[fixed_param]*tf.ones((1,M), dtype=tf.float64);
            elif (fixed_param == 'b_S'):
                b_S = self.fixed_params[fixed_param]*tf.ones((1,M), dtype=tf.float64);
            elif (fixed_param == 'b_V'):
                b_V = self.fixed_params[fixed_param]*tf.ones((1,M), dtype=tf.float64);

            elif (fixed_param == 'h_FFE'):
                h_FFE = self.fixed_params[fixed_param]*tf.ones((1,M), dtype=tf.float64);
            elif (fixed_param == 'h_FFP'):
                h_FFP = self.fixed_params[fixed_param]*tf.ones((1,M), dtype=tf.float64);

            elif (fixed_param == 'h_LATE'):
                h_LATE = self.fixed_params[fixed_param]*tf.ones((1,M), dtype=tf.float64);
            elif (fixed_param == 'h_LATP'):
                h_LATP = self.fixed_params[fixed_param]*tf.ones((1,M), dtype=tf.float64);
            elif (fixed_param == 'h_LATS'):
                h_LATS = self.fixed_params[fixed_param]*tf.ones((1,M), dtype=tf.float64);
            elif (fixed_param == 'h_LATV'):
                h_LATV = self.fixed_params[fixed_param]*tf.ones((1,M), dtype=tf.float64);

            elif (fixed_param == 'h_RUNE'):
                h_RUNE = self.fixed_params[fixed_param]*tf.ones((1,M), dtype=tf.float64);
            elif (fixed_param == 'h_RUNP'):
                h_RUNP = self.fixed_params[fixed_param]*tf.ones((1,M), dtype=tf.float64);
            elif (fixed_param == 'h_RUNS'):
                h_RUNS = self.fixed_params[fixed_param]*tf.ones((1,M), dtype=tf.float64);
            elif (fixed_param == 'h_RUNV'):
                h_RUNV = self.fixed_params[fixed_param]*tf.ones((1,M), dtype=tf.float64);

            elif (fixed_param == 'tau'):
                tau = self.fixed_params[fixed_param]*tf.ones((self.C,M), dtype=tf.float64);
            elif (fixed_param == 'n'):
                n = self.fixed_params[fixed_param]*tf.ones((self.C,M), dtype=tf.float64);
            elif (fixed_param == 's_0'):
                s_0 = self.fixed_params[fixed_param]*tf.ones((1,M), dtype=tf.float64);
            elif (fixed_param == 'a'):
                a = self.fixed_params[fixed_param]*tf.ones((1,M), dtype=tf.float64);
            elif (fixed_param == 'c_50'):
                c_50 = self.fixed_params[fixed_param]*tf.ones((1,M), dtype=tf.float64);

            else:
                print('Error: unknown fixed parameter: %s.' % fixed_param)
                raise NotImplementedError()

            ind += 1

        # Gather weights into the dynamics matrix W [K,M,4,4]
        W_EX = tf.stack([W_EE, -W_EP, -W_ES, tf.zeros((self.C,M), dtype=tf.float64)], axis=2);
        W_PX = tf.stack([W_PE, -W_PP, -W_PS, tf.zeros((self.C,M), dtype=tf.float64)], axis=2);
        W_SX = tf.stack([W_SE, tf.zeros((self.C,M), dtype=tf.float64), tf.zeros((self.C,M), dtype=tf.float64), -W_SV], axis=2);
        W_VX = tf.stack([W_VE, -W_VP, -W_VS, tf.zeros((self.C,M), dtype=tf.float64)], axis=2);
        W = tf.stack([W_EX, W_PX, W_SX, W_VX], axis=2); 

        # Gather inputs into b [K,M,4,1]
        b = tf.expand_dims(tf.stack([b_E, b_P, b_S, b_V], axis=2), 3); 
        h_FF = tf.expand_dims(tf.stack([h_FFE, h_FFP, tf.zeros((1,M), dtype=tf.float64), tf.zeros((1,M), dtype=tf.float64)], axis=2), 3);
        h_LAT = tf.expand_dims(tf.stack([h_LATE, h_LATP, h_LATS, h_LATV], axis=2), 3);
        h_RUN = tf.expand_dims(tf.stack([h_RUNE, h_RUNP, h_RUNS, h_RUNV], axis=2), 3); 

        # tau [K,M,1,1]
        tau = tf.expand_dims(tf.expand_dims(tau, 2), 3)
        # dynamics power [K,M,1,1]
        n = tf.expand_dims(tf.expand_dims(n, 2), 3)
        # reference stimulus [K,M,1,1]
        s_0 = tf.expand_dims(tf.expand_dims(s_0, 2), 3)

        if (self.model_opts['g_LAT'] == 'saturate'):
            # saturation shape [K,M,1,1]
            a = tf.expand_dims(tf.expand_dims(a, 2), 3)
            # 50% constrast value [K,M,1,1]
            c_50 = tf.expand_dims(tf.expand_dims(c_50, 2), 3)
        else:
            a = None
            c_50 = None


        return W, b, h_FF, h_LAT, h_RUN, tau, n, s_0, a, c_50

    def compute_h(self, b, h_FF, h_LAT, h_RUN, s_0, a=None, c_50=None):
        num_c = self.behavior['c_vals'].shape[0]
        num_s = self.behavior['s_vals'].shape[0]
        num_r = self.behavior['r_vals'].shape[0]
        hs = []
        for i in range(num_c):
            c = self.behavior['c_vals'][i]
            # compute g_FF for this condition
            if (self.model_opts['g_FF'] == 'c'):
                g_FF = c
            elif (self.model_opts['g_FF'] == 'saturate'):
                g_FF = tf.divide(tf.pow(c, a), tf.pow(c_50, a) + tf.pow(c, a))
            else:
                raise NotImplementedError()

            for j in range(num_s):
                s = self.behavior['s_vals'][j]
                # compute g_LAT for this condition
                if (self.model_opts['g_LAT'] == 'linear'):
                    g_LAT = tf.multiply(c, tf.nn.relu(s - s_0))
                elif (self.model_opts['g_LAT'] == 'square'):
                    g_LAT = tf.multiply(c, tf.nn.relu(tf.square(s) - tf.square(s_0)))
                else:
                    raise NotImplementedError()

                for k in range(num_r):
                    if (self.model_opts['g_RUN'] == 'r'):
                        r = self.behavior['r_vals'][k]
                    else:
                        raise NotImplementedError()

                    g_RUN = r
                    h_csr = b + tf.multiply(g_FF, h_FF) \
                              + tf.multiply(g_LAT, h_LAT) \
                              + tf.multiply(g_RUN, h_RUN)
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
        r0 = tf.constant(np.expand_dims(np.expand_dims(self.init_conds, 0), 0), dtype=tf.float64);
        r0 = tf.tile(r0, [self.C,M,1,1]); # [K,M,4,1]

        # construct the input
        def f(r, t):
            drdt = tf.divide(-r + tf.pow(tf.nn.relu(tf.matmul(W, r) + h), n), tau)
            return tf.clip_by_value(drdt, -1e6, 1e6);

        # time axis
        t = np.arange(0, self.T*self.dt, self.dt)

        # simulate ODE
        r_t = tf.contrib.integrate.odeint_fixed(f, r0, t, method='rk4')
        return r_t;
    
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

        if (self.behavior['type'] in ['data', 'difference']):
            T_x = self.simulation_suff_stats(z)
        else:
            raise NotImplementedError();
        
        return T_x

    def simulation_suff_stats(self, z):
        """Compute sufficient statistics that require simulation.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            T_x (tf.tensor): Simulation-derived sufficient statistics of samples.

        """

        #r1_t, r2_t = self.simulate(z);
        r_t = self.simulate(z); # [T, C, M, D, 1]

        if (self.behavior['type'] == 'difference'):
            diff_inds = self.behavior['diff_inds']
            r1_ss_list = []
            r2_ss_list = []
            for ind in range(len(diff_inds)):
                r1_ss_list.append(r_t[-1,0,:,ind,0])
                r2_ss_list.append(r_t[-1,1,:,ind,0])
            r1_ss = tf.stack(r1_ss_list, axis=1)
            r2_ss = tf.stack(r2_ss_list, axis=1)
            diff_ss = tf.expand_dims(r2_ss - r1_ss, 0)
            T_x = tf.concat((diff_ss, tf.square(diff_ss)), 2);

        elif (self.behavior['type'] == 'data'):
            r_shape = tf.shape(r_t)
            M = tf.shape[2]
            r_ss = tf.transpose(r1_t[-1,:,:,:,0], [1,2,0]) # [M,C,D];
            r_ss = tf.reshape(r_ss, [M, self.C*self.D])
            T_x = tf.concat((r_ss, tf.square(r_ss)), 2);

        return T_x

    def compute_mu(self,):
        """Calculate expected moment constraints given system paramterization.

        # Returns
            mu (np.array): Expected moment constraints.

        """

        if (self.behavior['type'] == 'difference'): 
            means = self.behavior['d_mean']
            variances = self.behavior["d_var"]
        elif (self.behavior['type'] == 'data'):
            means = np.reshape(self.behavior['r_mean'], (self.C*self.D,))
            variances = np.reshape(self.behavior['r_var'], (self.C*self.D,))
        first_moments = means
        second_moments = np.square(means) + variances
        mu = np.concatenate((first_moments, second_moments), axis=0)
        return mu

    def support_mapping(self, inputs):
        """TODO add documentation

        """
        return SoftPlusFlow([], inputs)

        


class R1RNN_input(system):
    """Rank-1 RNN with bistable states for low input magnitudes
	   See Fig. 2F - Mastrogiuseppe et. al. 2018

	# Attributes
		T (int): Number of consistency equation solve steps.
        Ics_0 (np.array): A set of initial conditions.
        Ics_1 (np.array): Another set of initial conditions.
		behavior_str (str): Determines sufficient statistics that characterize system.
	"""

    def __init__(self, T, Ics_0, Ics_1, behavior_str):
        self.behavior_str = behavior_str
        self.name = "R1RNN_input"
        self.D = 4
        self.eps = 0.8
        self.g = 0.8
        self.T = T
        self.Ics_0 = Ics_0
        self.Ics_1 = Ics_1
        self.num_suff_stats = 8

    def compute_suff_stats(self, z):
        """Compute sufficient statistics of density network samples.

		# Arguments
			z (tf.tensor): Density network system parameter samples.

		# Returns
			T_x (tf.tensor): Sufficient statistics of samples.
		"""
        if self.behavior_str == "bistable":
            T_x = self.simulation_suff_stats(z)
        else:
            raise NotImplementedError
        return T_x

    def simulation_suff_stats(self, z):
        """Compute sufficient statistics that require simulation.

		# Arguments
			z (tf.tensor): Density network system parameter samples.

		# Returns
			T_x (tf.tensor): Simulation-derived sufficient statistics of samples.

		"""

        if self.behavior_str == "bistable":
            sol = self.solve(z)
            sol_shape = tf.shape(sol)
            K = sol_shape[1]
            M = sol_shape[2]
            D = sol_shape[3]
            X = tf.clip_by_value(sol[2, :, :, :], -1e3, 1e3)
            X = tf.expand_dims(tf.reshape(tf.transpose(X, [1, 0, 2]), [M, K * D]), 0)
            T_x = tf.concat((X, tf.square(X)), 2)
        return T_x

    def solve(self, z):
        """Solve the consistency equations given parameters z.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            g(z) (tf.tensor): Simulated system activity.

        """

        Mm_tf = z[:, :, 0, :]
        Mn_tf = z[:, :, 1, :]
        Sim_tf = z[:, :, 2, :]
        Sin_tf = z[:, :, 3, :]

        Mm_tf = tf.tile(Mm_tf, [2, 1, 2])
        Mn_tf = tf.tile(Mn_tf, [2, 1, 2])
        Sim_tf = tf.tile(Sim_tf, [2, 1, 2])
        Sin_tf = tf.tile(Sin_tf, [2, 1, 2])

        # Mm = 3.5      # Mean of m
        # Mn = 1.       # Mean of n
        Mi = 0.0  # Mean of I

        # Sim = 1.      # Std of m
        # Sin = 1.      # Std of n
        Sip = 1.0

        Sini = np.concatenate(
            (0.5 * np.ones((1, 1, 2)), 1.0 * np.ones((1, 1, 2))), axis=0
        )

        def consistent_solve(y, g, eps, T):
            y_1 = y[:, :, :, 0]
            y_2 = y[:, :, :, 1]
            y_3 = y[:, :, :, 2]
            for i in range(T):
                Sii = tf.sqrt((Sini / Sin_tf) ** 2 + Sip ** 2)

                mu = Mm_tf * y_3 + Mi
                new1 = g * g * tf_integrals.PhiSq(mu, y_2) + Sim_tf ** 2 * y_3 ** 2
                new1 = new1 + Sii ** 2
                new2 = Mn_tf * tf_integrals.Phi(mu, y_2) + Sini * tf_integrals.Prime(
                    mu, y_2
                )

                y_new_1 = Mm_tf * new2 + Mi
                y_new_2 = (1 - eps) * y_2 + eps * new1
                y_new_3 = (1 - eps) * y_3 + eps * new2

                y_1 = y_new_1
                y_2 = y_new_2
                y_3 = y_new_3

            y_out = tf.stack([y_1, y_2, y_3], axis=0)
            return y_out

        Ics = np.concatenate(
            (np.expand_dims(self.Ics_0, 2), np.expand_dims(self.Ics_1, 2)), axis=2
        )
        Ics = np.tile(Ics, [2, 1, 1, 1])
        sol = consistent_solve(Ics, self.g, self.eps, self.T)

        out = sol
        return sol

    def compute_mu(self, behavior):
        """Calculate expected moment constraints given system paramterization.

        # Arguments
            behavior (dict): Parameterization of desired system behavior.

        # Returns
            mu (np.array): Expected moment constraints.

        """
        mu = behavior["mu"]
        Sigma = behavior["Sigma"]
        mu_mu = mu
        mu_Sigma = np.square(mu_mu) + Sigma
        mu = np.concatenate((mu_mu, mu_Sigma), axis=0)
        return mu

    def support_mapping(self, inputs):
        """TODO add documentation

        """
        return SoftPlusFlow([], inputs)


class R1RNN_GNG(system):
    """Rank-1 RNN for the Go No-Go task
       See Fig. 3 - Mastrogiuseppe et. al. 2018

    # Attributes
        T (int): Number of consistency equation solve steps.
        Ics_0 (np.array): A set of initial conditions.
        Ics_1 (np.array): Another set of initial conditions.
        behavior_str (str): Determines sufficient statistics that characterize system.
    """

    def __init__(self, T, Ics_0, behavior_str):
        self.behavior_str = behavior_str
        self.name = "R1RNN_GNG"
        self.D = 5
        self.eps = 0.8
        self.T = T
        self.Ics_0 = Ics_0
        self.num_suff_stats = 4;

    def compute_suff_stats(self, z):
        """Compute sufficient statistics of density network samples.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            T_x (tf.tensor): Sufficient statistics of samples.
        """
        if self.behavior_str == "gng":
            T_x = self.simulation_suff_stats(z)
        else:
            raise NotImplementedError
        return T_x

    def simulation_suff_stats(self, z):
        """Compute sufficient statistics that require simulation.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            T_x (tf.tensor): Simulation-derived sufficient statistics of samples.

        """

        if self.behavior_str == "gng":
            sol = self.solve(z)
            print(sol.shape);
            sol_shape = tf.shape(sol)
            K = sol_shape[1]
            M = sol_shape[2]
            D = sol_shape[3]
            print('sol shape');
            print(sol);
            X = tf.clip_by_value(sol[2, :, :, :], -1e3, 1e3)
            X = tf.expand_dims(tf.reshape(tf.transpose(X, [1, 0, 2]), [M, K * D]), 0)
            print('X shape');
            print(X);
            T_x = tf.concat((X, tf.square(X)), 2)
        return T_x

    def solve(self, z):
        """Solve the consistency equations given parameters z.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            g(z) (tf.tensor): Simulated system activity.

        """

        Mm_tf = z[:, :, 0, :]
        Mn_tf = z[:, :, 1, :]
        Sim_tf = z[:, :, 2, :]
        Sin_tf = z[:, :, 3, :]
        g_tf = z[:, :, 4, :];

        Mm_tf = tf.tile(Mm_tf, [2, 1, 1])
        Mn_tf = tf.tile(Mn_tf, [2, 1, 1])
        Sim_tf = tf.tile(Sim_tf, [2, 1, 1])
        Sin_tf = tf.tile(Sin_tf, [2, 1, 1])
        g_tf = tf.tile(g_tf, [2, 1, 1])

        Mi = 0.0  # Mean of I
        Sip = 1.0

        Sini = np.concatenate(
            (0.0 * np.ones((1, 1, 1)), 1.0 * np.ones((1, 1, 1))), axis=0
        )

        def consistent_solve(y, eps, T):
            y_1 = y[:, :, :, 0]
            y_2 = y[:, :, :, 1]
            y_3 = y[:, :, :, 2]
            for i in range(T):
                Sii = tf.sqrt((Sini / Sin_tf) ** 2 + Sip ** 2)

                mu = Mm_tf * y_3 + Mi
                new1 = tf.square(g_tf) * tf_integrals.PhiSq(mu, y_2) + Sim_tf ** 2 * y_3 ** 2
                new1 = new1 + Sii ** 2
                new2 = Mn_tf * tf_integrals.Phi(mu, y_2) + Sini * tf_integrals.Prime(
                    mu, y_2
                )

                y_new_1 = Mm_tf * new2 + Mi
                y_new_2 = (1 - eps) * y_2 + eps * new1
                y_new_3 = (1 - eps) * y_3 + eps * new2

                y_1 = y_new_1
                y_2 = y_new_2
                y_3 = y_new_3

            y_out = tf.stack([y_1, y_2, y_3], axis=0)
            return y_out

        Ics = np.expand_dims(self.Ics_0, 2);
        Ics = np.tile(Ics, [2, 1, 1, 1])
        sol = consistent_solve(Ics, self.eps, self.T)

        return sol

    def compute_mu(self, behavior):
        """Calculate expected moment constraints given system paramterization.

        # Arguments
            behavior (dict): Parameterization of desired system behavior.

        # Returns
            mu (np.array): Expected moment constraints.

        """
        mu = behavior["mu"]
        Sigma = behavior["Sigma"]
        mu_mu = mu
        mu_Sigma = np.square(mu_mu) + Sigma
        mu = np.concatenate((mu_mu, mu_Sigma), axis=0)
        return mu

    def map_to_parameter_support(self, layers, num_theta_params):
        """Augment density network with bijective mapping to support.

        # Arguments
            layers (list): List of ordered normalizing flow layers.
            num_theta_params (int): Running count of density network parameters.

        # Returns
            layers (list): layers augmented with final support mapping layer.
            num_theta_params (int): Updated count of density network parameters.
            
        """

        support_layer = SoftPlusLayer()
        num_theta_params += count_layer_params(support_layer)
        layers.append(support_layer)
        return layers, num_theta_params






class damped_harmonic_oscillator(system):
    """Damped harmonic oscillator.  Solution should be a line with noise.

    # Attributes
        D (int): parametric dimensionality
        T (int): number of time points
        dt (float): time resolution of simulation
        behavior_str (str): determines sufficient statistics that characterize system
    """

    def __init__(self, behavior_str, T, dt, init_conds, bounds):
        super().__init__(behavior_str)
        self.T = T
        self.dt = dt
        self.name = "damped_harmonic_oscillator"
        self.D = 3
        self.init_conds = init_conds
        self.num_suff_stats = 2 * T
        self.bounds = bounds

    def simulate(self, z):
        """Simulate the damped harmonic oscillator given parameters z.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            g(z) (tf.tensor): Simulated system activity.

        """
        z_shape = tf.shape(z)
        K = z_shape[0]
        M = z_shape[1]

        k = z[:, :, 0, :]
        c = z[:, :, 1, :]
        m = z[:, :, 2, :]

        w_0 = tf.sqrt(tf.divide(k, m))
        zeta = tf.divide(c, 2.0 * tf.sqrt(tf.multiply(m, k)))

        def dydt(y, t):
            y1 = y[0]
            y2 = y[1]

            y1_dot = y2
            y2_dot = -2.0 * tf.multiply(tf.multiply(w_0, zeta), y2) - tf.multiply(
                tf.square(w_0), y1
            )

            ydot = tf.stack([y1_dot, y2_dot])
            return ydot

        y0 = tf.concat(
            (
                self.init_conds[0] * tf.ones((1, K, M, 1), dtype=tf.float64),
                self.init_conds[1] * tf.ones((1, K, M, 1), dtype=tf.float64),
            ),
            axis=0,
        )
        t = np.linspace(0, self.dt * (self.T - 1), self.T)

        out = tf.contrib.integrate.odeint_fixed(dydt, y0, t, method="rk4")

        return tf.transpose(out[:, :, :, :, 0], [2, 3, 1, 0])
        # make it K x M x D (sys) x T

    def compute_suff_stats(self, z):
        """Compute sufficient statistics of density network samples.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            T_x (tf.tensor): Sufficient statistics of samples.

        """

        if self.behavior_str == "trajectory":
            T_x = self.simulation_suff_stats(z)
        else:
            raise NotImplementedError
        return T_x

    def simulation_suff_stats(self, z):
        """Compute sufficient statistics that require simulation.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            T_x (tf.tensor): Simulation-derived sufficient statistics of samples.

        """

        if self.behavior_str == "trajectory":
            XY = self.simulate(z)
            X = tf.clip_by_value(XY[:, :, 0, :], -1e3, 1e3)
            T_x = tf.concat((X, tf.square(X)), 2)
        return T_x

    def compute_mu(self, behavior):
        """Calculate expected moment constraints given system paramterization.

        # Arguments
            behavior (dict): Parameterization of desired system behavior.

        # Returns
            mu (np.array): Expected moment constraints.

        """

        mu = behavior["mu"]
        Sigma = behavior["Sigma"]
        mu_mu = mu
        mu_Sigma = np.square(mu_mu) + Sigma
        print(mu_mu.shape, mu_Sigma.shape)
        mu = np.concatenate((mu_mu, mu_Sigma), 0)
        return mu

    def map_to_parameter_support(self, layers, num_theta_params):
        """Augment density network with bijective mapping to parameter support.

        # Arguments
            layers (list): List of ordered normalizing flow layers.
            num_theta_params (int): Running count of density network parameters.

        # Returns
            layers (list): layers augmented with final support mapping layer.
            num_theta_params (int): Updated count of density network parameters.

        """

        support_layer = IntervalFlowLayer(
            "IntervalFlowLayer", self.bounds[0], self.bounds[1]
        )
        num_theta_params += count_layer_params(support_layer)
        layers.append(support_layer)
        return layers, num_theta_params



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

