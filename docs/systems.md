---
title: Systems
permalink: /systems/
---

<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML' async></script>

<div class="topnav">
  <a class="active" href="../">Home</a>
  <a href="#">DSN Systems Library</a>
</div>

Many commonly used models in theoretical neuroscience are already implemented as built-in system classes in the DSN library.

# dsn.util.systems


*****
## <a name="system"> </a> system
```python
system(self, behavior_str)
```
Base class for systems using DSN modeling.

Degenerate solution networks (DSNs) learn the full parameter space of models
given some model behavioral constraints.  Given some system choice and a
behavioral specification, these classes are designed to perform the system
specific functions that are necessary for training the corresponding DSN.

__Attributes__

- `behavior_str (str)`: Determines sufficient statistics that characterize system.


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
system.compute_mu(self, behavior)
```
Calculate expected moment constraints given system paramterization.

__Arguments__

- __behavior (dict)__: Parameterization of desired system behavior.

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
system.center_suff_stats_by_mu(self, T_x, mu)
```
Center sufficient statistics by the mean parameters mu.

__Arguments__

- __T_x (tf.tensor)__: Sufficient statistics of samples.
- __mu (np.array)__: mean vector of constraints

__Returns__

`T_x_mu_centered (tf.tensor)`: Mean centered sufficient statistics of samples.



*****
## <a name="linear_2D"> </a> linear\_2D
```python
linear_2D(self, behavior_str)
```
Linear two-dimensional system.

This is a simple system explored in the <a href="../#linear_2D_example">DSN tutorial</a>, which demonstrates the
utility of DSNs in an intuitive way.

\begin{equation}
\dot{x} = Ax, A = \begin{bmatrix} a_1 & a_2 \\\\ a_3 & a_4 \end{bmatrix}
\end{equation}

Behaviors:

'oscillation' - specify a distribution of oscillatory frequencies

__Attributes__

- `behavior_str (str)`: In `['oscillation']`.  Determines sufficient statistics that characterize system.

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
E_{x\sim p(x \mid z)}\left[T(x)\right] = f_{p,T}(z) = E \begin{bmatrix} \text{real}(\lambda_1) \\\\ \text{real}(\lambda_1)^2 \\\\ \text{imag}(\lambda_1) \\\\ \text{imag}(\lambda_1)^2 \end{bmatrix}
\end{equation}

__Arguments__

- __z (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Sufficient statistics of samples.


### compute\_mu
```python
linear_2D.compute_mu(self, behavior)
```
Calculate expected moment constraints given system paramterization.

__Arguments__

- __behavior (dict)__: Parameterization of desired system behavior.

__Returns__

`mu (np.array)`: Expected moment constraints.



*****
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



*****
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



*****
## <a name="V1_circuit"> </a> V1\_circuit
```python
V1_circuit(self, behavior_str, param_str, T, dt, init_conds)
```
4-neuron V1 circuit.

This is the standard 4-neuron rate model of V1 activity consisting of
 - E: pyramidal (excitatory) neurons
 - P: parvalbumim expressing inhibitory neurons
 - S: somatostatin expressing inhibitory neurons
 - V: vasoactive intestinal peptide expressing inhibitory neurons

dr/dt = -r + [Wr + h]+^n

__Attributes__

- `behavior_str (str)`:
- `'ss'`: steady state responses
- `param-str (str)`:
- `'h'`: Learn the input parameters.
- `'W'`: Learn the connectivity parameters.
- `T (int)`: Number of simulation time points.
- `dt (float)`: Time resolution of simulation.
- `init_conds (list)`: Specifies the initial state of the system.

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
V1_circuit.compute_mu(self, behavior)
```
Calculate expected moment constraints given system paramterization.

__Arguments__

- __behavior (dict)__: Parameterization of desired system behavior.

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



*****
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


