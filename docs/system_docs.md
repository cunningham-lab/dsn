# dsn.util.systems

## system
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


### map_to_parameter_support
```python
system.map_to_parameter_support(self, layers, num_theta_params)
```
Augment density network with bijective mapping to parameter support.
### compute_suff_stats
```python
system.compute_suff_stats(self, phi)
```
Compute sufficient statistics of density network samples.
### analytic_suff_stats
```python
system.analytic_suff_stats(self, phi)
```
Compute closed form sufficient statistics.
### simulation_suff_stats
```python
system.simulation_suff_stats(self, phi)
```
Compute sufficient statistics that require simulation.
### compute_mu
```python
system.compute_mu(self, behavior)
```
Calculate expected moment constraints given system paramterization.
### center_suff_stats_by_mu
```python
system.center_suff_stats_by_mu(self, T_x, mu)
```
Center sufficient statistics by the mean parameters mu.
## linear_2D
```python
linear_2D(self, behavior_str)
```
Linear two-dimensional system.

__Attributes__

- `behavior_str (str)`: Determines sufficient statistics that characterize system.

### compute_suff_stats
```python
linear_2D.compute_suff_stats(self, phi)
```
Compute sufficient statistics of density network samples.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Sufficient statistics of samples.

### analytic_suff_stats
```python
linear_2D.analytic_suff_stats(self, phi)
```
Compute closed form sufficient statistics.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Analytic sufficient statistics of samples.


### compute_mu
```python
linear_2D.compute_mu(self, behavior)
```
Calculate expected moment constraints given system paramterization.

__Arguments__

- __behavior (dict)__: Parameterization of desired system behavior.

__Returns__

`mu (np.array)`: Expected moment constraints.


## R1RNN_input
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

### compute_suff_stats
```python
R1RNN_input.compute_suff_stats(self, phi)
```
Compute sufficient statistics of density network samples.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Sufficient statistics of samples.

### simulation_suff_stats
```python
R1RNN_input.simulation_suff_stats(self, phi)
```
Compute sufficient statistics that require simulation.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Simulation-derived sufficient statistics of samples.


### solve
```python
R1RNN_input.solve(self, phi)
```
Solve the consistency equations given parameters phi.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`g(phi) (tf.tensor)`: Simulated system activity.


### compute_mu
```python
R1RNN_input.compute_mu(self, behavior)
```
Calculate expected moment constraints given system paramterization.

__Arguments__

- __behavior (dict)__: Parameterization of desired system behavior.

__Returns__

`mu (np.array)`: Expected moment constraints.


### map_to_parameter_support
```python
R1RNN_input.map_to_parameter_support(self, layers, num_theta_params)
```
Augment density network with bijective mapping to support.

__Arguments__

- __layers (list)__: List of ordered normalizing flow layers.
- __num_theta_params (int)__: Running count of density network parameters.

__Returns__

`layers (list)`: layers augmented with final support mapping layer.
`num_theta_params (int)`: Updated count of density network parameters.


## R1RNN_GNG
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

### compute_suff_stats
```python
R1RNN_GNG.compute_suff_stats(self, phi)
```
Compute sufficient statistics of density network samples.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Sufficient statistics of samples.

### simulation_suff_stats
```python
R1RNN_GNG.simulation_suff_stats(self, phi)
```
Compute sufficient statistics that require simulation.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Simulation-derived sufficient statistics of samples.


### solve
```python
R1RNN_GNG.solve(self, phi)
```
Solve the consistency equations given parameters phi.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`g(phi) (tf.tensor)`: Simulated system activity.


### compute_mu
```python
R1RNN_GNG.compute_mu(self, behavior)
```
Calculate expected moment constraints given system paramterization.

__Arguments__

- __behavior (dict)__: Parameterization of desired system behavior.

__Returns__

`mu (np.array)`: Expected moment constraints.


### map_to_parameter_support
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


## V1_circuit
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
V1_circuit.simulate(self, phi)
```
Simulate the V1 4-neuron circuit given parameters phi.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`g(phi) (tf.tensor)`: Simulated system activity.


### compute_suff_stats
```python
V1_circuit.compute_suff_stats(self, phi)
```
Compute sufficient statistics of density network samples.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Sufficient statistics of samples.


### simulation_suff_stats
```python
V1_circuit.simulation_suff_stats(self, phi)
```
Compute sufficient statistics that require simulation.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Simulation-derived sufficient statistics of samples.


### compute_mu
```python
V1_circuit.compute_mu(self, behavior)
```
Calculate expected moment constraints given system paramterization.

__Arguments__

- __behavior (dict)__: Parameterization of desired system behavior.

__Returns__

`mu (np.array)`: Expected moment constraints.


### map_to_parameter_support
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


## damped_harmonic_oscillator
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
damped_harmonic_oscillator.simulate(self, phi)
```
Simulate the damped harmonic oscillator given parameters phi.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`g(phi) (tf.tensor)`: Simulated system activity.


### compute_suff_stats
```python
damped_harmonic_oscillator.compute_suff_stats(self, phi)
```
Compute sufficient statistics of density network samples.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Sufficient statistics of samples.


### simulation_suff_stats
```python
damped_harmonic_oscillator.simulation_suff_stats(self, phi)
```
Compute sufficient statistics that require simulation.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Simulation-derived sufficient statistics of samples.


### compute_mu
```python
damped_harmonic_oscillator.compute_mu(self, behavior)
```
Calculate expected moment constraints given system paramterization.

__Arguments__

- __behavior (dict)__: Parameterization of desired system behavior.

__Returns__

`mu (np.array)`: Expected moment constraints.


### map_to_parameter_support
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


## null_on_interval
```python
null_on_interval(self, D, a=0, b=1)
```
Null system.  D parameters no constraints. Solution should be uniform on interval.

__Attributes__

- `D (int)`: Parametric dimensionality.
- `a (float)`: Beginning of interval.
- `b (float)`: End of interval.

### compute_suff_stats
```python
null_on_interval.compute_suff_stats(self, phi)
```
Compute sufficient statistics of density network samples.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Sufficient statistics of samples.


### compute_mu
```python
null_on_interval.compute_mu(self, behavior)
```
Calculate expected moment constraints given system paramterization.

__Arguments__

- __behavior (dict)__: Parameterization of desired system behavior.

__Returns__

`mu (np.array)`: Expected moment constraints.


### map_to_parameter_support
```python
null_on_interval.map_to_parameter_support(self, layers, num_theta_params)
```
Augment density network with bijective mapping to parameter support.

__Arguments__

- __layers (list)__: List of ordered normalizing flow layers.
- __num_theta_params (int)__: Running count of density network parameters.

__Returns__

`layers (list)`: layers augmented with final support mapping layer.
`num_theta_params (int)`: Updated count of density network parameters.


## one_con_on_interval
```python
one_con_on_interval(self, D, a=0, b=1)
```
System with one constraint.  D parameters dim1 == dim2.
Solution should be uniform on the plane.

__Attributes__

- `D (int)`: Parametric dimensionality.
- `a (float)`: Beginning of interval.
- `b (float)`: End of interval.

### compute_suff_stats
```python
one_con_on_interval.compute_suff_stats(self, phi)
```
Compute sufficient statistics of density network samples.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Sufficient statistics of samples.


### compute_mu
```python
one_con_on_interval.compute_mu(self, behavior)
```
Calculate expected moment constraints given system paramterization.

__Arguments__

- __behavior (dict)__: Parameterization of desired system behavior.

__Returns__

`mu (np.array)`: Expected moment constraints.


### map_to_parameter_support
```python
one_con_on_interval.map_to_parameter_support(self, layers, num_theta_params)
```
Augment density network with bijective mapping to parameter support.

__Arguments__

- __layers (list)__: List of ordered normalizing flow layers.
- __num_theta_params (int)__: Running count of density network parameters.

__Returns__

`layers (list)`: layers augmented with final support mapping layer.
`num_theta_params (int)`: Updated count of density network parameters.


## two_con_on_interval
```python
two_con_on_interval(self, D, a=0, b=1)
```
System with two constraints.  D parameters dim1 == dim2, dim2 == dim3
Solution should be uniform on the plane.

__Attributes__

- `D (int)`: Parametric dimensionality.
- `a (float)`: Beginning of interval.
- `b (float)`: End of interval.

### compute_suff_stats
```python
two_con_on_interval.compute_suff_stats(self, phi)
```
Compute sufficient statistics of density network samples.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Sufficient statistics of samples.


### compute_mu
```python
two_con_on_interval.compute_mu(self, behavior)
```
Calculate expected moment constraints given system paramterization.

__Arguments__

- __behavior (dict)__: Parameterization of desired system behavior.

__Returns__

`mu (np.array)`: Expected moment constraints.


### map_to_parameter_support
```python
two_con_on_interval.map_to_parameter_support(self, layers, num_theta_params)
```
Augment density network with bijective mapping to parameter support.

__Arguments__

- __layers (list)__: List of ordered normalizing flow layers.
- __num_theta_params (int)__: Running count of density network parameters.

__Returns__

`layers (list)`: layers augmented with final support mapping layer.
`num_theta_params (int)`: Updated count of density network parameters.


## MultivariateNormal
```python
MultivariateNormal(self, behavior_str='moments', D=2)
```
Multivariate Gaussian validation system.

__Attributes__

- `behavior_str (str)`: Determines sufficient statistics that characterize system.
- `D (int)`: Parametric dimensionality.


### compute_suff_stats
```python
MultivariateNormal.compute_suff_stats(self, phi)
```
Compute sufficient statistics of density network samples.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Sufficient statistics of samples.


### analytic_suff_stats
```python
MultivariateNormal.analytic_suff_stats(self, phi)
```
Compute closed form sufficient statistics.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Analytic sufficient statistics of samples.

### compute_mu
```python
MultivariateNormal.compute_mu(self, behavior)
```
Compute the mean parameterization (mu) given the mean parameters.

__Arguments__

- __behavior (dict)__: Mean parameters of behavioral distribution.

__Returns__

`mu (np.array)`: The mean parameterization vector of the exponential family.


## linear_1D
```python
linear_1D(self, behavior_str, T, dt, init_conds)
```
Exponential growth/decay.

__Attributes__

- `D (int)`: parametric dimensionality
- `T (int)`: number of time points
- `dt (float)`: time resolution of simulation
- `behavior_str (str)`: determines sufficient statistics that characterize system

### compute_suff_stats
```python
linear_1D.compute_suff_stats(self, phi)
```
Compute sufficient statistics of density network samples.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Sufficient statistics of samples.

### analytic_suff_stats
```python
linear_1D.analytic_suff_stats(self, phi)
```
Compute closed form sufficient statistics.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Analytic sufficient statistics of samples.

### simulation_suff_stats
```python
linear_1D.simulation_suff_stats(self, phi)
```
Compute sufficient statistics that require simulation.

__Arguments__

- __phi (tf.tensor)__: Density network system parameter samples.

__Returns__

`T_x (tf.tensor)`: Simulation-derived sufficient statistics of samples.

### compute_mu
```python
linear_1D.compute_mu(self, behavior)
```
Calculate expected moment constraints given system paramterization.

__Arguments__

- __behavior (dict)__: Parameterization of desired system behavior.

__Returns__

`mu (np.array)`: Expected moment constraints.


