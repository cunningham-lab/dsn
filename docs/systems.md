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


## <a name="Linear2D"> </a> Linear2D
```python
Linear2D(self, fixed_params, behavior)
```
Linear two-dimensional system.

This is a simple system explored in the <a href="../#Linear2D_example">DSN tutorial</a>, which demonstrates the
utility of DSNs in an intuitive way.

\begin{equation}
\tau \dot{x} = Ax, A = \begin{bmatrix} a_1 & a_2 \\\\ a_3 & a_4 \end{bmatrix}
\end{equation}

Behaviors:

'oscillation' - specify a distribution of oscillatory frequencies

__Attributes__

- `behavior (dict)`: see Linear2D.compute_suff_stats

### get\_all\_sys\_params
```python
Linear2D.get_all_sys_params(self)
```
Returns ordered list of all system parameters and individual element labels.

- $$A$$ - 2x2 dynamics matrix
- $$\tau$$ - scalar timescale parameter

__Returns__

`all_params (list)`: List of strings of all parameters of full system model.
`all_param_labels (list)`: List of tex strings for all parameters.

### get\_T\_x\_labels
```python
Linear2D.get_T_x_labels(self)
```
Returns `T_x_labels`.

Behaviors:

'oscillation' - $$[$$real($$\lambda_1$$), $$\frac{\text{imag}(\lambda_1)}{2 \pi}$$, real$$(\lambda_1)^2$$, $$(\frac{\text{imag}(\lambda_1)}{2 \pi})^2]$$

__Returns__

`T_x_labels (list)`: List of tex strings for elements of $$T(x)$$.


### compute\_suff\_stats
```python
Linear2D.compute_suff_stats(self, z)
```
Compute sufficient statistics of density network samples.

Behaviors:

'oscillation' - Specifies a distribution of oscillatory frequencies and
                expansion/decay factors using the eigendecomposition of
                the dynamics matrix.
\begin{equation}
E_{x\sim p(x \mid z)}\left[T(x)\right] = f_{p,T}(z) = E \begin{bmatrix} \text{real}(\lambda_1) \\\\ \frac{\text{imag}(\lambda_1)}{2\pi} \\\\ \text{real}(\lambda_1)^2 \\\\ (\frac{\text{imag}(\lambda_1)}{2\pi}^2 \end{bmatrix}
\end{equation}

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Sufficient statistics of samples.


### compute\_mu
```python
Linear2D.compute_mu(self)
```
Calculate expected moment constraints given system paramterization.

__Returns__

`mu (np.array)`: Expected moment constraints.


## <a name="V1Circuit"> </a> V1Circuit
```python
V1Circuit(self, fixed_params, behavior, model_opts={'g_FF': 'c', 'g_LAT': 'linear', 'g_RUN': 'r'}, T=40, dt=0.25, init_conds=array([[1. ],
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

__Attributes__

- `behavior (dict)`: see V1Circuit.compute_suff_stats
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
V1Circuit.get_all_sys_params(self)
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
V1Circuit.get_T_x_labels(self)
```
Returns `T_x_labels`.

Behaviors:

'difference' - $$[d_{E,ss}, d_{P,ss}, d_{S,ss}, d_{V,ss}, d_{E,ss}^2, d_{P,ss}^2, d_{S,ss}^2, d_{V,ss}^2]$$

'data' - $$[r_{E,ss}(c,s,r), ...,  r_{E,ss}(c,s,r)^2, ...]$$

__Returns__

`T_x_labels (list)`: List of tex strings for elements of $$T(x)$$.


### filter\_Z
```python
V1Circuit.filter_Z(self, z)
```
Returns the system matrix/vector variables depending free parameter ordering.

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`W (tf.tensor)`: [C,M,4,4] Dynamics matrices.
`b (tf.tensor)`: [1,M,4,1] Static inputs.
`h_FF (tf.tensor)`: [1,M,4,1] Feed forward inputs.
`h_LAT (tf.tensor)`: [1,M,4,1] Lateral inputs.
`h_RUN (tf.tensor)`: [1,M,4,1] Running inputs.
`tau (tf.tensor)`: [C,M,1,1] Dynamics timescales.
`n (tf.tensor)`: [C,M,1,1] Dynamics power coefficients.
`s_0 (tf.tensor)`: [1,M,1,1] Reference stimulus values.
`a (tf.tensor)`: [1,M,1,1] Contrast saturation shape.
`c_50 (tf.tensor)`: [1,M,1,1] Contrast at 50%.


### simulate
```python
V1Circuit.simulate(self, z)
```
Simulate the V1 4-neuron circuit given parameters z.

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`g(z) (tf.tensor)`: Simulated system activity.


### compute\_suff\_stats
```python
V1Circuit.compute_suff_stats(self, z)
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
V1Circuit.simulation_suff_stats(self, z)
```
Compute sufficient statistics that require simulation.

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Simulation-derived sufficient statistics of samples.


### compute\_mu
```python
V1Circuit.compute_mu(self)
```
Calculate expected moment constraints given system paramterization.

__Returns__

`mu (np.array)`: Expected moment constraints.


### support\_mapping
```python
V1Circuit.support_mapping(self, inputs)
```
Maps from real numbers to support of parameters.

__Arguments:__

    inputs (np.array): Input from previous layers of the DSN.

__Returns__

`Z (np.array)`: Samples from the DSN at the final layer.

## <a name="LowRankRNN"> </a> LowRankRNN
```python
LowRankRNN(self, fixed_params, behavior, model_opts={'rank': 1, 'input_type': 'spont'}, solve_its=25, solve_eps=0.8)
```
Recent work by ([Matrogiusseppe & Ostojic, 2018](#Mastrogiuseppe2018Linking)) allows us to
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
$$\chi_{ij} \sim \mathcal{N}(0, \frac{1}{N})$$, and the structured
component is a sum of $$r$$ unit rank terms:

$$P_{ij} = \sum_{k=1}^r \frac{m_i^{(k)}n_j^{(k)}}{N}$$

The nonlinearity $$\phi$$ is set to $$tanh$$ in this software, but
the theory is general for many other activation functions.


__Attributes__

- `behavior (dict)`: see LowRankRNN.compute_suff_stats
- `model_opts (dict)`:
  * model_opts[`'rank'`]
    * `1` (default) Rank 1 network
    * `r` any other pos int (TODO)
  * model_opts[`'input_type'`]
    * `'spont'` (default) No input.
- `solve_its (int)`: Number of langevin dynamics simulation steps.
- `solve_eps (float)`: Langevin dynamics solver step-size.

### get\_all\_sys\_params
```python
LowRankRNN.get_all_sys_params(self)
```
Returns ordered list of all system parameters and individual element labels.

When `model_opts['rank'] == 1`

 - $$g$$ - strength of the random matrix component
 - $$M_m$$ - mean value of right connectivity vector
 - $$M_n$$ - mean value of left connectivity vector
 - $$\Sigma_m$$ - variance of values in right connectivity vector

When `model_opts['rank'] == 2`

 - TODO

__Returns__

`all_params (list)`: List of strings of all parameters of full system model.
`all_param_labels (list)`: List of tex strings for all parameters.

### get\_T\_x\_labels
```python
LowRankRNN.get_T_x_labels(self)
```
Returns `T_x_labels`.

Behaviors:

'struct_chaos' - $$[\mu, \Delta_{\infty}, (\Delta_0 - \Delta_{\infty}), \mu^2, \Delta_{\infty}^2, (\Delta_0 - \Delta_{\infty})^2]$$

__Returns__

`T_x_labels (list)`: List of tex strings for elements of $$T(x)$$.


### filter\_Z
```python
LowRankRNN.filter_Z(self, z)
```
Returns the system matrix/vector variables depending free parameter ordering.

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`g (tf.tensor)`: [1,M] Strength of the random matrix component.
`Mm (tf.tensor)`: [1,M] Mean value of right connectivity vector.
`Mn (tf.tensor)`: [1,M] Mean value of left connectivity vector.
`Sm (tf.tensor)`: [1,M] Variance of values in right connectivity vector.


### compute\_suff\_stats
```python
LowRankRNN.compute_suff_stats(self, z)
```
Compute sufficient statistics of density network samples.

Behaviors:

'struct_chaos' -

  When `model_opts['rank'] == 1` and `model_opts['input_type'] == 'spont'`

  Set constraints on the mean unit activity $$\mu$$, the static variance
  $$\Delta_{\infty}$$, and the temporal variance $$\Delta_T = \Delta_0 - \Delta_{\infty}$$.
  $$\mu$$, $$\Delta_0$$, and $$\Delta_{\infty}$$ can be found for a rank-1
  no-input network by solving the following consistency equations.

  $$\mu = F(\mu, \Delta_0, \Delta_\infty) = M_m M_n \int \mathcal{D}z \phi(\mu + \sqrt{\Delta_0} z)$$

  $$\Delta_0 = G(\mu, \Delta_0, \Delta_\infty) = [\Delta_\infty^2 + 2g^2\{\int \mathcal{D}z \Phi^2(\mu + \sqrt{\Delta_0}z) - \int \mathcal{D}z [\int \mathcal{D}x \Phi(\mu + \sqrt{\Delta_0 - \Delta_\infty}x $$
  $$ + \sqrt{\Delta_\infty}z)]^2\} +M_n^2 \Sigma_m^2 \langle[\phi_i]\rangle^2(\Delta_0 - \Delta_\infty)]^{\frac{1}{2}} $$

  $$\Delta_\infty = H(\mu, \Delta_0, \Delta_\infty) = g^2 \int \mathcal{D}z \left[ \int \mathcal{D}x \Phi(\mu + \sqrt{\Delta_0 - \Delta_\infty} + \sqrt{\Delta_\infty}z \right]^2 + M_n^2 \Sigma_m^2 \langle [\phi_i] \rangle^2$$

  The solutions are found via a Langevin dynamics simulation with step size
  `self.solve_eps` and number of iterations `self.solve_its`.

  $$\dot{\mu} = -\mu + F(\mu, \Delta_0, \Delta_\infty)$$

  $$\dot{\Delta_0} = \Delta_0 + G(\mu, \Delta_0, \Delta_\infty)$$

  $$\dot{\Delta_\infty} = -\Delta_\infty + H(\mu, \Delta_0, \Delta_\infty)$$

  The total constraint vector is
  \begin{equation}
  E_{x\sim p(x \mid z)}\left[T(x)\right] = \begin{bmatrix} \mu \\\\ \Delta_\infty \\\\ \Delta_0 - \Delta_\infty \\\\ \mu \\\\ \Delta_\infty^2 \\\\ (\Delta_0 - \Delta_\infty)^2 \end{bmatrix}
  \end{equation}

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Sufficient statistics of samples.


### compute\_mu
```python
LowRankRNN.compute_mu(self)
```
Calculate expected moment constraints given system paramterization.

__Returns__

`mu (np.array)`: Expected moment constraints.


### support\_mapping
```python
LowRankRNN.support_mapping(self, inputs)
```
Maps from real numbers to support of parameters.

__Arguments:__

    inputs (np.array): Input from previous layers of the DSN.

__Returns__

`Z (np.array)`: Samples from the DSN at the final layer.



# References #

Dipoppa, Mario, et al. *[Vision and locomotion shape the interactions between neuron types in mouse visual cortex](https://www.sciencedirect.com/science/article/pii/S0896627318302435){:target="_blank"}*. Neuron 98.3 (2018): 602-615. <a name="Dipoppa2018Vision"></a>

Pfeffer, Carsten K., et al. [Inhibition of inhibition in visual cortex: the logic of connections between molecularly distinct interneurons](https://www.nature.com/articles/nn.3446){:target="_blank"}*." Nature neuroscience 16.8 (2013): 1068. <a name="Pfeffer2013Inhibition"></a>

Mastrogiuseppe, Francesca, and Srdjan Ostojic. *[Linking connectivity, dynamics, and computations in low-rank recurrent neural networks](https://www.sciencedirect.com/science/article/pii/S0896627318305439){:target="_blank"}*. Neuron 99.3 (2018): 609-623. <a name="Mastrogiuseppe2018Linking"></a>


