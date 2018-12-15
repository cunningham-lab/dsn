---
title: Systems
permalink: /systems/
---

<div class="topnav">
  <a class="active" href="../">Home</a>
  <a href="#">DSN Systems Library</a>
</div>

# Built-in systems library #

Many commonly used models in theoretical neuroscience are already implemented as built-in system classes in the DSN library.  For an introduction to the system class, see section X.X.X of the [tutorial](index.md).

Checking with-in page linking
[dev install](index.md#dev-install). 

*****

## 2-D linear system ##
Linear two-dimensional system

### Attributes ###
**behavior_str** (str): *Determines sufficient statistics that characterize system.*

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