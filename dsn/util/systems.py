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
        behavior_str (str): Determines sufficient statistics that characterize system.

        all_param_labels (list): List of tex strings for all parameters.
        z_labels (list): List of tex strings for free parameters.
        T_x_labels (list): List of tex strings for elements of $$T(x)$$.
    """

    def __init__(self, fixed_params, behavior_str):
        """System constructor.

		# Arguments 
            fixed_params (dict): Specifies fixed parameters and their values.
			behavior_str (str): Determines sufficient statistics that characterize system.
	
		"""
        self.fixed_params = fixed_params
        self.behavior_str = behavior_str
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

    def compute_mu(self, behavior):
        """Calculate expected moment constraints given system paramterization.

        # Arguments
            behavior (dict): Parameterization of desired system behavior.

        # Returns
            mu (np.array): Expected moment constraints.

        """
        raise NotImplementedError()

    def map_to_parameter_support(self, layers, num_theta_params):
        """Augment density network with bijective mapping to parameter support.

        # Arguments
            layers (list): List of ordered normalizing flow layers.
            num_theta_params (int): Running count of density network parameters.

        # Returns
            layers (list): layers augmented with final support mapping layer. \\\\
            num_theta_params (int): Updated count of density network parameters.

        """
        return layers, num_theta_params

    def center_suff_stats_by_mu(self, T_x, mu):
        """Center sufficient statistics by the mean parameters mu.
    
        # Arguments
            T_x (tf.tensor): Sufficient statistics of samples.
            mu (np.array): mean vector of constraints

        # Returns
            T_x_mu_centered (tf.tensor): Mean centered sufficient statistics of samples.
        
        """
        return T_x - np.expand_dims(np.expand_dims(mu, 0), 1)


class linear_2D(system):
    """Linear two-dimensional system.

    This is a simple system explored in the <a href="../#linear_2D_example">DSN tutorial</a>, which demonstrates the
    utility of DSNs in an intuitive way.  

    \\begin{equation}
    \dot{x} = Ax, A = \\begin{bmatrix} a_1 & a_2 \\\\\\\\ a_3 & a_4 \end{bmatrix}
    \end{equation}

    Behaviors:

    'oscillation' - specify a distribution of oscillatory frequencies

    # Attributes
        behavior_str (str): In `['oscillation']`.  Determines sufficient statistics that characterize system.
    """

    def __init__(self, fixed_params, behavior_str):
        super().__init__(fixed_params, behavior_str)
        self.name = "linear_2D"

    def get_all_sys_params(self,):
        all_params = ['A', 'tau'];
        all_param_labels = {'A':[r'$a_1$', r'$a_2$', r'$a_3$', r'$a_4$'], \
                            'tau':[r'$\tau$']}
        return all_params, all_param_labels

    def get_T_x_labels(self,):
        if (self.behavior_str == 'oscillation'):
            T_x_labels = ['1', '2', '3', '4'];
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
        E_{x\\sim p(x \\mid z)}\\left[T(x)\\right] = f_{p,T}(z) = E \\begin{bmatrix} \\text{real}(\\lambda_1) \\\\\\\\ \\text{real}(\\lambda_1)^2 \\\\\\\\ \\text{imag}(\\lambda_1) \\\\\\\\ \\text{imag}(\\lambda_1)^2 \end{bmatrix}
        \end{equation}

		# Arguments
			z (tf.tensor): Density network system parameter samples.

		# Returns
			T_x (tf.tensor): Sufficient statistics of samples.

		"""
        if self.behavior_str == "oscillation":
            z_shape = tf.shape(z)
            K = z_shape[0]
            M = z_shape[1]

            a1 = z[:, :, 0, :]
            a2 = z[:, :, 1, :]
            a3 = z[:, :, 2, :]
            a4 = z[:, :, 3, :]

            beta = tf.complex(tf.square(a1 + a4) - 4 * (a1 * a4 + a2 * a3), np.float64(0.0))
            beta_sqrt = tf.sqrt(beta)
            real_common = tf.complex(0.5 * (a1 + a4), np.float64(0.0))

            lambda_1 = real_common + 0.5 * beta_sqrt
            lambda_1_real = tf.real(lambda_1)
            lambda_1_imag = tf.imag(lambda_1)
            moments = [
                lambda_1_real,
                tf.square(lambda_1_real),
                lambda_1_imag,
                tf.square(lambda_1_imag),
            ]
            T_x = tf.concat(moments, 2)
        else:
            raise NotImplementedError
        return T_x

    def compute_mu(self, behavior):
        """Calculate expected moment constraints given system paramterization.

        # Arguments
            behavior (dict): Parameterization of desired system behavior.

        # Returns
            mu (np.array): Expected moment constraints.

        """
        means = behavior["means"]
        variances = behavior["variances"]
        first_moments = means
        second_moments = np.square(means) + variances
        mu = np.array(
            [first_moments[0], second_moments[0], first_moments[1], second_moments[1]]
        )
        return mu


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

    def map_to_parameter_support(self, layers, num_theta_params):
        """Augment density network with bijective mapping to support.

		# Arguments
			layers (list): List of ordered normalizing flow layers.
			num_theta_params (int): Running count of density network parameters.

		# Returns
			(list): layers augmented with final support mapping layer.
			(int): Updated count of density network parameters.
            
		"""

        support_layer = SoftPlusLayer()
        num_theta_params += count_layer_params(support_layer)
        layers.append(support_layer)
        return layers, num_theta_params


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


class V1_circuit(system):
    """ 4-neuron V1 circuit.

        This is the standard 4-neuron rate model of V1 activity consisting of 
         - E: pyramidal (excitatory) neurons
         - P: parvalbumim expressing inhibitory neurons
         - S: somatostatin expressing inhibitory neurons
         - V: vasoactive intestinal peptide expressing inhibitory neurons

        dr/dt = -r + [Wr + h]+^n

    # Attributes
        behavior_str (str):
            'ss': steady state responses
        param-str (str):
            'h': Learn the input parameters.
            'W': Learn the connectivity parameters.
        T (int): Number of simulation time points.
        dt (float): Time resolution of simulation.
        init_conds (list): Specifies the initial state of the system.
    """

    def __init__(self, behavior_str, param_str, T, dt, init_conds):
        super().__init__(behavior_str)
        self.param_str = param_str
        self.T = T
        self.dt = dt
        self.name = "V1_circuit"
        # determine dimensionality and number of constraints
        self.D = 0;
        self.num_suff_stats = 0;

        self.z_labels = []
        self.T_x_labels = []

        if (behavior_str in ['ss_all']):
            if (param_str in ['h', 'both']):
                self.D += 8;
                self.z_labels += [r'$h_{E,1}$', r'$h_{P,1}$', r'$h_{S,1}$', r'$h_{V,1}$', \
                                r'$h_{E,2}$', r'$h_{P,2}$', r'$h_{S,2}$', r'$h_{V,2}$']
            if (param_str == ['W', 'both']):
                self.D += 11;
                raise NotImplementedError();
            self.num_suff_stats += 8;
            self.T_x_labels += [r'$d_{E,ss}$', r'$d_{P,ss}$', r'$d_{S,ss}$', r'$d_{V,ss}$', \
                                  r'$d_{E,ss}^2$', r'$d_{P,ss}^2$', r'$d_{S,ss}^2$', r'$d_{V,ss}^2$']

        elif (behavior_str in ['ss_SV']):
            if (param_str in ['h', 'both']):
                self.D += 8;
                self.z_labels += [r'$h_{E,1}$', r'$h_{P,1}$', r'$h_{S,1}$', r'$h_{V,1}$', \
                                r'$h_{E,2}$', r'$h_{P,2}$', r'$h_{S,2}$', r'$h_{V,2}$']
            if (param_str == ['W', 'both']):
                self.D += 11;
                raise NotImplementedError();
            self.num_suff_stats += 4;
            self.T_x_labels += [r'$d_{S,ss}$', r'$d_{V,ss}$', \
                                  r'$d_{S,ss}^2$', r'$d_{V,ss}^2$']

        else:
            raise NotImplementedError();

        self.init_conds = init_conds

    def simulate(self, z):
        """Simulate the V1 4-neuron circuit given parameters z.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            g(z) (tf.tensor): Simulated system activity.

        """
        # remove trailing dimension
        z = z[:,:,:,0];
        z_shape = tf.shape(z)
        K = z_shape[0]
        M = z_shape[1]

        ind = 0;
        if (self.behavior_str in ['ss_all', 'ss_SV']):
            if (self.param_str in ['h', 'both']):
                h1 = tf.expand_dims(z[:,:,0:4], 3);
                h2 = tf.expand_dims(z[:,:,4:8], 3);
                ind += 8;
            else:
                # this would have to be based on a hypothesized input structure
                h1 = np.ones((4,1));
                h2 = -np.ones((4,1));
                raise NotImplementedError();

            if (self.param_str in ['W', 'both']):
                W_EE = z[:,:,ind];
                W_EP = z[:,:,ind+1];
                W_ES = z[:,:,ind+2];
                W_EX = tf.stack([W_EE, -W_EP, -W_ES, tf.zeros((K,M))], axis=2);

                W_PE = z[:,:,ind+3];
                W_PP = z[:,:,ind+4];
                W_PS = z[:,:,ind+5];
                W_PX = tf.stack([W_PE, -W_PP, -W_PS, tf.zeros((K,M))], axis=2);

                W_SE = z[:,:,ind+6];
                W_SV = z[:,:,ind+7];
                W_SX = tf.stack([W_SE, tf.zeros((K,M)), tf.zeros((K,M)), -W_SV], axis=2);

                W_VE = z[:,:,ind+8];
                W_VP = z[:,:,ind+9];
                W_VS = z[:,:,ind+10];
                W_VX = tf.stack([W_VE, -W_VP, -W_VS, tf.zeros((K,M))], axis=2);

                W = tf.stack([W_EX, W_PX, W_SX, W_VX], axis=2);

            else:
                # Using values from Pfeffer et al. 2013
                E_pre = 1.0;
                W_EE = E_pre;
                W_EP = 1.0;
                W_ES = 0.54;

                W_PE = E_pre;
                W_PP = 1.01;
                W_PS = 0.33;

                W_SE = E_pre;
                W_SV = 0.15;

                W_VE = E_pre;
                W_VP = 0.22;
                W_VS = 0.77;

                W = np.array([[W_EE, -W_EP, -W_ES,   0.0], \
                   [W_PE, -W_PP, -W_PS,   0.0], \
                   [W_SE,   0.0,   0.0, -W_SV], \
                   [W_VE, -W_VP, -W_VS,   0.0]]);
                W = np.expand_dims(np.expand_dims(W, 0), 0);
                W = tf.constant(W);
                W = tf.tile(W, [K,M,1,1]);

            # initial conditions
            r0 = tf.constant(np.expand_dims(np.expand_dims(self.init_conds, 0), 0));
            r0 = tf.tile(r0, [K,M,1,1]);

            # transition function
            n = 2.0;
            def f1(r, t):
                drdt = -r + tf.pow(tf.nn.relu(tf.matmul(W, r) + h1), n);
                return tf.clip_by_value(drdt, 1e-3, 1e3);

            def f2(r, t):
                drdt = -r + tf.pow(tf.nn.relu(tf.matmul(W, r) + h2), n);
                return tf.clip_by_value(drdt, 1e-3, 1e3);

            # time axis
            t = np.arange(0,self.T*self.dt, self.dt);

            r1_t = tf.contrib.integrate.odeint_fixed(f1, r0, t, method='rk4')
            r2_t = tf.contrib.integrate.odeint_fixed(f2, r0, t, method='rk4')

            return [r1_t, r2_t];
        
        else:
            raise NotImplementedError();

    def compute_suff_stats(self, z):
        """Compute sufficient statistics of density network samples.

        # Arguments
            z (tf.tensor): Density network system parameter samples.

        # Returns
            T_x (tf.tensor): Sufficient statistics of samples.

        """

        if (self.behavior_str in ["ss_all", "ss_SV"]):
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

        r1_t, r2_t = self.simulate(z);

        if (self.behavior_str in ["ss_all"]):
            r1_ss = r1_t[-1,:,:,:,0];
            r2_ss = r2_t[-1,:,:,:,0];
        elif (self.behavior_str in ["ss_SV"]):
            # extract somatostatin and VIP responses
            r1_ss = r1_t[-1,:,:,2:,0];  
            r2_ss = r2_t[-1,:,:,2:,0];
        diff_ss = r2_ss - r1_ss;
        T_x = tf.concat((diff_ss, tf.square(diff_ss)), 2);
        print(r1_ss.shape, diff_ss.shape, T_x.shape);
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
        mu = np.concatenate((mu_mu, mu_Sigma), 0)
        return mu

    def mu_to_ellipse(self, behavior):
        mu = behavior["mu"]
        Sigma = behavior["Sigma"]

    def map_to_parameter_support(self, layers, num_theta_params):
        """Augment density network with bijective mapping to parameter support.

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
    if system_str in ["linear_2D"]:
        return linear_2D
    elif system_str in ["damped_harmonic_oscillator", "dho"]:
        return damped_harmonic_oscillator
    elif system_str in ["rank1_rnn"]:
        return RNN_rank1
    elif system_str in ["R1RNN_input"]:
        return R1RNN_input
    elif system_str in ["R1RNN_GNG"]:
        return R1RNN_GNG
    elif system_str in ["V1_circuit"]:
        return V1_circuit

