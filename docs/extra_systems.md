---
title: Systems
permalink: /systems/
---

<div class="topnav">
  <a class="active" href="../">Home</a>
  <a href="#">DSN Systems Library</a>
</div>

Many commonly used models in theoretical neuroscience are already implemented as built-in system classes in the DSN library. 

# system #
Base class for systems using DSN modeling.

Degenerate solution networks (DSNs) learn the full parameter space of models
given some model behavioral constraints. Given some system choice and a
behavioral specification, these classes are designed to perform the system
specific functions that are necessary for training the corresponding DSN.

**Attributes**:\\
**behavior_str** (str): *Determines sufficient statistics that characterize system.*


*****

# <a name="linear_2D"> </a> linear_2D #
Linear two-dimensional system.

**Attributes**:\\
**behavior_str** (str): *Determines sufficient statistics that characterize system.*

### analytic_suff_stats ###
Compute closed form sufficient statistics.

**Args**:\\
**phi** (tf.tensor): *Density network system parameter samples.*

**Returns**:\\
**T_x** (tf.tensor): *Analytic sufficient statistics of samples.*




### center_suff_stats_by_mu ###
Center sufficient statistics by the mean parameters mu.


### compute_mu ###
Calculate expected moment constraints given system paramterization.

**Args**:\\
**behavior** (dict): *Parameterization of desired system behavior.*

**Returns**:\\
**mu** (np.array): *Expected moment constraints.*




### compute_suff_stats ###
Compute sufficient statistics of density network samples.

**Args**:\\
**phi** (tf.tensor): *Density network system parameter samples.*

**Returns**:\\
**T_x** (tf.tensor): *Sufficient statistics of samples.*



### map_to_parameter_support ###
Augment density network with bijective mapping to parameter support.


### simulation_suff_stats ###
Compute sufficient statistics that require simulation.


# V1_circuit #
 4-neuron V1 circuit.

This is the standard 4-neuron rate model of V1 activity consisting of
- E: pyramidal (excitatory) neurons
- P: parvalbumim expressing inhibitory neurons
- S: somatostatin expressing inhibitory neurons
- V: vasoactive intestinal peptide expressing inhibitory neurons

dr/dt = -r + [Wr + h]+^n

**Attributes**:\\
**behavior_str** (str): *(str):*
'ss': steady state responses
**param-str** (str): *(str):*
'h': Learn the input parameters.
'W': Learn the connectivity parameters.
**T** (int): *Number of simulation time points.*
**dt** (float): *Time resolution of simulation.*
**init_conds** (list): *Specifies the initial state of the system.*

### analytic_suff_stats ###
Compute closed form sufficient statistics.


### center_suff_stats_by_mu ###
Center sufficient statistics by the mean parameters mu.


### compute_mu ###
Calculate expected moment constraints given system paramterization.

**Args**:\\
**behavior** (dict): *Parameterization of desired system behavior.*

**Returns**:\\
**mu** (np.array): *Expected moment constraints.*




### compute_suff_stats ###
Compute sufficient statistics of density network samples.

**Args**:\\
**phi** (tf.tensor): *Density network system parameter samples.*

**Returns**:\\
**T_x** (tf.tensor): *Sufficient statistics of samples.*




### map_to_parameter_support ###
Augment density network with bijective mapping to parameter support.

**Args**:\\
**layers** (list): *List of ordered normalizing flow layers.*
**num_theta_params** (int): *Running count of density network parameters.*

**Returns**:\\
**layers** (list): *layers augmented with final support mapping layer.*
**num_theta_params** (int): *Updated count of density network parameters.*




### simulate ###
Simulate the V1 4-neuron circuit given parameters phi.

**Args**:\\
**phi** (tf.tensor): *Density network system parameter samples.*

**Returns**:\\
**g(phi)** (tf.tensor): *Simulated system activity.*




### simulation_suff_stats ###
Compute sufficient statistics that require simulation.

**Args**:\\
**phi** (tf.tensor): *Density network system parameter samples.*

**Returns**:\\
**T_x** (tf.tensor): *Simulation-derived sufficient statistics of samples.*




*****


### Behaviors ###
#### Stability ####
Blah blah blah

#### Oscillations ####
Blah blah blah


*****

## 4-neuron V1 circuit ##
insert image of circuit


*****

## Low-rank recurrent neural networks #

Recent work by (Matroguisseppe & Ostojic, 2018) allows us to derive statistical properties of the behavior of recurrent neural networks (RNNs) given a low-rank parameterization of their connectivity.  This work builds on dynamic mean field theory (DMFT) for neural networks (Sompolinsky et al. 1988), which is exact in the limit of infinite neurons, but has been shown to yield accurate approximations for finite size networks.  We provide some brief background regarding DMFT and the recent theoretical advancements that facilitate our examination of the solution space of RNNs performing computations.  A description of the supported RNN parameterizations and types of behaviors follows.

## Dynamic mean field theory ##
Mean field theory (MFT) originated as a useful tool for physicists studying many-body problems, particularly interactions of many particles in proximity.  Deriving an equation for the probability of configurations of such systems of particles in equilibrium requires a partition function, which is essentially the normalizing constant of the probability distribution.  The partition function relies on the Hamiltonian of the system, which is an expression for the total energy of the system.  Many body problems in physics are usually pairwise interaction models, resulting in  combinatoric growth issue in the calculation of the Hamiltonian.  A mean field assumption that some degrees of the freedom of the system have independent probabilities makes approximations to the Hamilton tractable.  Importantly, when minimizing the free energy of the system (to find the equilibrium state), the mean field assumption allows the derivation of consistency equations.  For a given system parameterization, we can solve the consistency equations using an off-the-shelf nonlinear system of equations solver.

Using the same modeling strategy as MFT, physicists developed dynamic mean field theory (DMFT) to describe interactions in spin glasses. Later, this same formalism was used to describe dynamic properties of unstructured neural networks (Somp ’88).

Latex equations:

<img src="https://latex.codecogs.com/gif.latex?\dot{x}_i(t)&space;=&space;-x_i(t)&space;&plus;&space;\sum_{j=1}^N&space;J_{ij}&space;\phi(x_j(t))" title="\dot{x}_i(t) = -x_i(t) + \sum_{j=1}^N J_{ij} \phi(x_j(t))" />

text in the middle

<img src="https://latex.codecogs.com/gif.latex?\dot{x}_i(t)&space;=&space;-x_i(t)&space;&plus;&space;\eta_i(t)" title="\dot{x}_i(t) = -x_i(t) + \eta_i(t)" />

more text in the middle

<img src="https://latex.codecogs.com/gif.latex?\langle&space;\eta_i(t)&space;\eta_i(t&space;&plus;&space;\tau)&space;\rangle&space;=&space;J^2&space;C(\tau)" title="\langle \eta_i(t) \eta_i(t + \tau) \rangle = J^2 C(\tau)" />

more text in the middle

<img src="https://latex.codecogs.com/gif.latex?C(\tau)&space;=&space;\left[&space;\frac{1}{N}&space;\sum_{j=1}^N&space;\phi_i(t)&space;\phi_i(t&plus;\tau)&space;\right]_J&space;=&space;\langle&space;\phi_i(t)&space;\phi(t&space;&plus;&space;\tau)&space;\rangle" title="C(\tau) = \left[ \frac{1}{N} \sum_{j=1}^N \phi_i(t) \phi_i(t+\tau) \right]_J = \langle \phi_i(t) \phi(t + \tau) \rangle" />

more text in the middle

<img src="https://latex.codecogs.com/gif.latex?x_i(t)&space;=&space;\int_{-\infty}^t&space;e^{t'-t}&space;\eta_i(t')&space;dt'" title="x_i(t) = \int_{-\infty}^t e^{t'-t} \eta_i(t') dt'" />