---
title: Systems
permalink: /systems/
---

<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML' async></script>

<div class="topnav">
  <a class="active" href="../">Home</a>
  <a href="#">DSN Systems Library</a>
</div>

Some neural circuit models from theoretical neuroscience are already implemented as built-in system classes in the DSN library.

# dsn.util.systems

## <a name="system"> </a> system
```python
system(self, fixed_params, behavior)
```
Base class for systems using DSN modeling.

Degenerate solution networks (DSNs) learn the full parameter space of models
given some model behavioral constraints.  Given some system choice and a
behavioral specification, these classes are designed to perform the system
specific functions that are necessary for training the corresponding DSN.

__Attributes__

- `self.D (int)`: Dimensionality of $$z$$.
- `self.num_suff_stats (int)`: Dimensionality of behavioral constraint vector
                               $$T(x)$$.
- `all_params (list)`: List of strings of all parameters of full system model.
- `fixed_params (dict)`: Parameter string indexes its fixed value.
- `free_params (list)`: List of strings in `all_params` but not `fixed_params.keys()`.
                        These params make up z.
- `behavior (dict)`: Contains the behavioral type and the constraints.
- `mu (np.array)`: The mean constrain vector for DSN optimization.
- `all_param_labels (list)`: List of tex strings for all parameters.
- `z_labels (list)`: List of tex strings for free parameters.
- `T_x_labels (list)`: List of tex strings for elements of $$T(x)$$.

### get\_all\_sys\_params
```python
system.get_all_sys_params(self)
```
Returns ordered list of all system parameters and individual element labels.

__Returns__

`all_params (list)`: List of strings of all parameters of full system model.
`all_param_labels (list)`: List of tex strings for all parameters.

### get\_free\_params
```python
system.get_free_params(self)
```
Returns members of `all_params` not in `fixed_params.keys()`.

__Returns__

`free_params (list)`: List of strings of parameters in $$z$$.


### get\_z\_labels
```python
system.get_z_labels(self)
```
Returns `z_labels`.

__Returns__

`z_labels (list)`: List of tex strings for free parameters.


### get\_T\_x\_labels
```python
system.get_T_x_labels(self)
```
Returns `T_x_labels`.

__Returns__

`T_x_labels (list)`: List of tex strings for elements of $$T(x)$$.


### compute\_suff\_stats
```python
system.compute_suff_stats(self, z)
```
Compute sufficient statistics of density network samples.

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Sufficient statistics of samples.


### compute\_mu
```python
system.compute_mu(self)
```
Calculate expected moment constraints given system paramterization.

__Returns__

`mu (np.array)`: Expected moment constraints.


### map\_to\_parameter\_support
```python
system.map_to_parameter_support(self, layers, num_theta_params)
```
Augment density network with bijective mapping to parameter support.

__Arguments__

- __layers (list)__: List of ordered normalizing flow layers.
- __num_theta_params (int)__: Running count of density network parameters.

__Returns__

`layers (list)`: layers augmented with final support mapping layer. \\
`num_theta_params (int)`: Updated count of density network parameters.


### center\_suff\_stats\_by\_mu
```python
system.center_suff_stats_by_mu(self, T_x)
```
Center sufficient statistics by the mean parameters mu.

__Arguments__

- __T_x (tf.tensor)__: Sufficient statistics of samples.
- __mu (np.array)__: mean vector of constraints

__Returns__

`T_x_mu_centered (tf.tensor)`: Mean centered sufficient statistics of samples.


## <a name="linear_2D"> </a> linear\_2D
```python
linear_2D(self, fixed_params, behavior)
```
Linear two-dimensional system.

This is a simple system explored in the <a href="../#linear_2D_example">DSN tutorial</a>, which demonstrates the
utility of DSNs in an intuitive way.

\begin{equation}
\tau \dot{x} = Ax, A = \begin{bmatrix} a_1 & a_2 \\\\ a_3 & a_4 \end{bmatrix}
\end{equation}

Behaviors:

'oscillation' - specify a distribution of oscillatory frequencies

__Attributes__

- `behavior (dict)`: see linear_2D.compute_suff_stats

### get\_all\_sys\_params
```python
linear_2D.get_all_sys_params(self)
```
Returns ordered list of all system parameters and individual element labels.

- $$A$$ - 2x2 dynamics matrix
- $$\tau$$ - scalar timescale parameter

__Returns__

`all_params (list)`: List of strings of all parameters of full system model.
`all_param_labels (list)`: List of tex strings for all parameters.

### get\_T\_x\_labels
```python
linear_2D.get_T_x_labels(self)
```
Returns `T_x_labels`.

Behaviors:

'oscillation' - $$[$$real($$\lambda_1$$), imag($$\lambda_1$$), real$$(\lambda_1)^2$$, imag$$(\lambda_1)^2]$$

__Returns__

`T_x_labels (list)`: List of tex strings for elements of $$T(x)$$.


### compute\_suff\_stats
```python
linear_2D.compute_suff_stats(self, z)
```
Compute sufficient statistics of density network samples.

Behaviors:

'oscillation' - Specifies a distribution of oscillatory frequencies and
                expansion/decay factors using the eigendecomposition of
                the dynamics matrix.
\begin{equation}
E_{x\sim p(x \mid z)}\left[T(x)\right] = f_{p,T}(z) = E \begin{bmatrix} \text{real}(\lambda_1) \\\\ \text{imag}(\lambda_1) \\\\ \text{real}(\lambda_1)^2 \\\\ \text{imag}(\lambda_1)^2 \end{bmatrix}
\end{equation}

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Sufficient statistics of samples.


### compute\_mu
```python
linear_2D.compute_mu(self)
```
Calculate expected moment constraints given system paramterization.

__Returns__

`mu (np.array)`: Expected moment constraints.


## <a name="V1_circuit"> </a> V1\_circuit
```python
V1_circuit(self, fixed_params, behavior, model_opts={'g_FF': 'c', 'g_LAT': 'linear', 'g_RUN': 'r'}, T=20, dt=0.25, init_conds=array([[1. ],
       [1.1],
       [1.2],
       [1.3]]))
```
4-neuron V1 circuit.

This is the standard 4-neuron rate model of V1 activity consisting of
 - E: pyramidal (excitatory) neurons
 - P: parvalbumim expressing inhibitory neurons
 - S: somatostatin expressing inhibitory neurons
 - V: vasoactive intestinal peptide (VIP) expressing inhibitory neurons

 [include a graphic of the circuit connectivity]

The dynamics of each neural populations average rate
$$r = \begin{bmatrix} r_E \\ r_P \\ r_S \\ r_V \end{bmatrix}$$
are given by:
\begin{equation}
\tau \frac{dr}{dt} = -r + [Wr + h]_+^n
\end{equation}

In some cases, these neuron types do not send projections to one of the other
types.  Additionally, much work has been done to measure the relative magnitudes
of the synaptic projections between neural types.
\begin{equation}
W = \begin{bmatrix} W_{EE} & -1.0 & -0.54 & 0 \\\\ W_{PE} & -1.01 & -0.33 & 0 \\\\ W_{SE} & 0 & 0 & -0.15 \\\\ W_{VE} & -0.22 & -0.77 & 0 \end{bmatrix}
\end{equation}

In this model, we are interested in capturing V1 responses across varying
contrasts $$c$$, stimulus sizes $$s$$, and locomotion $$r$$ conditions.

\begin{equation}
h = b + g_{FF}(c) h_{FF} + g_{LAT}(c,s) h_{LAT} + g_{RUN}(r) h_{RUN}
\end{equation}

\begin{equation} \begin{bmatrix} h_E \\\\ h_P \\\\ h_S \\\\ h_V \end{bmatrix}
 = \begin{bmatrix} b_E \\\\ b_P \\\\ b_S \\\\ b_V \end{bmatrix} + g_{FF}(c) \begin{bmatrix} h_{FF,E} \\\\ h_{FF,P} \\\\ 0 \\\\ 0 \end{bmatrix} + g_{LAT}(c,s) \begin{bmatrix} h_{LAT,E} \\\\ h_{LAT,P} \\\\ h_{LAT,S} \\\\ h_{LAT,V} \end{bmatrix} + g_{RUN}(r) \begin{bmatrix} h_{RUN,E} \\\\ h_{RUN,P} \\\\ h_{RUN,S} \\\\ h_{RUN,V} \end{bmatrix}
\end{equation}

where $$g_{FF}(c)$$, $$g_{LAT}(c,s)$$, and $$g_{FF}(r)$$ modulate the input
parmeterization $$h$$ according to condition.  See initialization argument
`model_opts` on how to set the form of these functions.

__Attributes__

- `behavior (dict)`: see V1_circuit.compute_suff_stats
- `model_opts (dict)`:
  * model_opts[`'g_FF'`]
    * `'c'` (default) $$g_{FF}(c) = c$$
    * `'saturate'` $$g_{FF}(c) = \frac{c^a}{c_{50}^a + c^a}$$
  * model_opts[`'g_LAT'`]
    * `'linear'` (default) $$g_{LAT}(c,s) = c[s_0 - s]_+$$
    * `'square'` $$g_{LAT}(c,s) = c[s_0^2 - s^2]_+$$
  * model_opts[`'g_RUN'`]
    * `'r'` (default) $$g_{RUN}(r) = r$$
- `T (int)`: Number of simulation time points.
- `dt (float)`: Time resolution of simulation.
- `init_conds (list)`: Specifies the initial state of the system.

### get\_all\_sys\_params
```python
V1_circuit.get_all_sys_params(self)
```
Returns ordered list of all system parameters and individual element labels.

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
- $$\tau$$ - dynamics timescale
- $$n$$ - scalar for power of dynamics
- $$s_0$$ - reference stimulus level

When `model_opts['g_FF'] == 'saturate'`
- $$a$$ - contrast saturation shape
- $$c_{50}$$ - constrast at 50%

__Returns__

`all_params (list)`: List of strings of all parameters of full system model.
`all_param_labels (list)`: List of tex strings for all parameters.

### get\_T\_x\_labels
```python
V1_circuit.get_T_x_labels(self)
```
Returns `T_x_labels`.

Behaviors:

'difference' - $$[d_{E,ss}, d_{P,ss}, d_{S,ss}, d_{V,ss}, d_{E,ss}^2, d_{P,ss}^2, d_{S,ss}^2, d_{V,ss}^2]$$

'data' - $$[r_{E,ss}(c,s,r), ...,  r_{E,ss}(c,s,r)^2, ...]$$

__Returns__

`T_x_labels (list)`: List of tex strings for elements of $$T(x)$$.


### simulate
```python
V1_circuit.simulate(self, z)
```
Simulate the V1 4-neuron circuit given parameters z.

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`g(z) (tf.tensor)`: Simulated system activity.


### compute\_suff\_stats
```python
V1_circuit.compute_suff_stats(self, z)
```
Compute sufficient statistics of density network samples.

Behaviors:

'difference' -

  The total number of conditions from all of
  self,behavior.c_vals, s_vals, and r_vals should be two.
  The steady state of the first condition $$(c_1,s_1,r_1)$$ is
  subtracted from that of the second condition $$(c_2,s_2,r_2)$$ to get a
  difference vector
  \begin{equation}
  d_{\alpha,ss} = r_{\alpha,ss}(c_2,s_2,r_2) - r_{\alpha,ss}(c_1,s_1,r_1)
  \end{equation}

  The total constraint vector is
  \begin{equation}
  E_{x\sim p(x \mid z)}\left[T(x)\right] = \begin{bmatrix} d_{E,ss} \\\\ d_{P,ss} \\\\ d_{S,ss} \\\\ d_{V,ss} \\\\ d_{E,ss}^2 \\\\ d_{P,ss}^2 \\\\ d_{S,ss}^2 \\\\ d_{V,ss}^2 \end{bmatrix}
  \end{equation}


'data' -

  The user specifies the grid inputs for conditions via
  self.behavior.c_vals, s_vals, and r_vals.  The first and second
  moments of the steady states for these conditions make up the
  sufficient statistics vector.  Since the index is $$(c,s,r)$$,
  values of r are iterated over first, then s, then c (as is
  the c-standard) to construct the $$T(x)$$ vector.

  The total constraint vector is
  \begin{equation}
  E_{x\sim p(x \mid z)}\left[T(x)\right] = \begin{bmatrix} r_{E,ss}(c,s,r) \\\\ ... \\\\  r_{E,ss}(c,s,r)^2 \\\\ ... \end{bmatrix}
  \end{equation}

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Sufficient statistics of samples.


### simulation\_suff\_stats
```python
V1_circuit.simulation_suff_stats(self, z)
```
Compute sufficient statistics that require simulation.

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Simulation-derived sufficient statistics of samples.


### compute\_mu
```python
V1_circuit.compute_mu(self)
```
Calculate expected moment constraints given system paramterization.

__Returns__

`mu (np.array)`: Expected moment constraints.


### map\_to\_parameter\_support
```python
V1_circuit.map_to_parameter_support(self, layers, num_theta_params)
```
Augment density network with bijective mapping to parameter support.

__Arguments__

- __layers (list)__: List of ordered normalizing flow layers.
- __num_theta_params (int)__: Running count of density network parameters.

__Returns__

`layers (list)`: layers augmented with final support mapping layer.
`num_theta_params (int)`: Updated count of density network parameters.


## <a name="R1RNN_input"> </a> R1RNN\_input
```python
R1RNN_input(self, T, Ics_0, Ics_1, behavior_str)
```
Rank-1 RNN with bistable states for low input magnitudes
See Fig. 2F - Mastrogiuseppe et. al. 2018

__Attributes__

- `T (int)`: Number of consistency equation solve steps.
- `Ics_0 (np.array)`: A set of initial conditions.
- `Ics_1 (np.array)`: Another set of initial conditions.
- `behavior_str (str)`: Determines sufficient statistics that characterize system.

### compute\_suff\_stats
```python
R1RNN_input.compute_suff_stats(self, z)
```
Compute sufficient statistics of density network samples.

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Sufficient statistics of samples.

### simulation\_suff\_stats
```python
R1RNN_input.simulation_suff_stats(self, z)
```
Compute sufficient statistics that require simulation.

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Simulation-derived sufficient statistics of samples.


### solve
```python
R1RNN_input.solve(self, z)
```
Solve the consistency equations given parameters z.

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`g(z) (tf.tensor)`: Simulated system activity.


### compute\_mu
```python
R1RNN_input.compute_mu(self, behavior)
```
Calculate expected moment constraints given system paramterization.

__Arguments__

- __behavior (dict)__: Parameterization of desired system behavior.

__Returns__

`mu (np.array)`: Expected moment constraints.


### map\_to\_parameter\_support
```python
R1RNN_input.map_to_parameter_support(self, layers, num_theta_params)
```
Augment density network with bijective mapping to support.

__Arguments__

- __layers (list)__: List of ordered normalizing flow layers.
- __num_theta_params (int)__: Running count of density network parameters.

__Returns__

`(list)`: layers augmented with final support mapping layer.
`(int)`: Updated count of density network parameters.


## <a name="R1RNN_GNG"> </a> R1RNN\_GNG
```python
R1RNN_GNG(self, T, Ics_0, behavior_str)
```
Rank-1 RNN for the Go No-Go task
See Fig. 3 - Mastrogiuseppe et. al. 2018

__Attributes__

- `T (int)`: Number of consistency equation solve steps.
- `Ics_0 (np.array)`: A set of initial conditions.
- `Ics_1 (np.array)`: Another set of initial conditions.
- `behavior_str (str)`: Determines sufficient statistics that characterize system.

### compute\_suff\_stats
```python
R1RNN_GNG.compute_suff_stats(self, z)
```
Compute sufficient statistics of density network samples.

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Sufficient statistics of samples.

### simulation\_suff\_stats
```python
R1RNN_GNG.simulation_suff_stats(self, z)
```
Compute sufficient statistics that require simulation.

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Simulation-derived sufficient statistics of samples.


### solve
```python
R1RNN_GNG.solve(self, z)
```
Solve the consistency equations given parameters z.

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`g(z) (tf.tensor)`: Simulated system activity.


### compute\_mu
```python
R1RNN_GNG.compute_mu(self, behavior)
```
Calculate expected moment constraints given system paramterization.

__Arguments__

- __behavior (dict)__: Parameterization of desired system behavior.

__Returns__

`mu (np.array)`: Expected moment constraints.


### map\_to\_parameter\_support
```python
R1RNN_GNG.map_to_parameter_support(self, layers, num_theta_params)
```
Augment density network with bijective mapping to support.

__Arguments__

- __layers (list)__: List of ordered normalizing flow layers.
- __num_theta_params (int)__: Running count of density network parameters.

__Returns__

`layers (list)`: layers augmented with final support mapping layer.
`num_theta_params (int)`: Updated count of density network parameters.


## <a name="damped_harmonic_oscillator"> </a> damped\_harmonic\_oscillator
```python
damped_harmonic_oscillator(self, behavior_str, T, dt, init_conds, bounds)
```
Damped harmonic oscillator.  Solution should be a line with noise.

__Attributes__

- `D (int)`: parametric dimensionality
- `T (int)`: number of time points
- `dt (float)`: time resolution of simulation
- `behavior_str (str)`: determines sufficient statistics that characterize system

### simulate
```python
damped_harmonic_oscillator.simulate(self, z)
```
Simulate the damped harmonic oscillator given parameters z.

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`g(z) (tf.tensor)`: Simulated system activity.


### compute\_suff\_stats
```python
damped_harmonic_oscillator.compute_suff_stats(self, z)
```
Compute sufficient statistics of density network samples.

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Sufficient statistics of samples.


### simulation\_suff\_stats
```python
damped_harmonic_oscillator.simulation_suff_stats(self, z)
```
Compute sufficient statistics that require simulation.

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Simulation-derived sufficient statistics of samples.


### compute\_mu
```python
damped_harmonic_oscillator.compute_mu(self, behavior)
```
Calculate expected moment constraints given system paramterization.

__Arguments__

- __behavior (dict)__: Parameterization of desired system behavior.

__Returns__

`mu (np.array)`: Expected moment constraints.


### map\_to\_parameter\_support
```python
damped_harmonic_oscillator.map_to_parameter_support(self, layers, num_theta_params)
```
Augment density network with bijective mapping to parameter support.

__Arguments__

- __layers (list)__: List of ordered normalizing flow layers.
- __num_theta_params (int)__: Running count of density network parameters.

__Returns__

`layers (list)`: layers augmented with final support mapping layer.
`num_theta_params (int)`: Updated count of density network parameters.


