---
layout: default
---
# Introduction #
Theoretical neuroscientists have complemented experimental investigations in our field by designing and testing mathematical models of neural activity (Abbott, 2008). Free of the burdens of experimental neuroscience, theorists do this with rapid turnover, frequently rejecting and sometimes promoting new models, which make accurate postdictions of established experiments.  A challenge in this scientific paradigm is the indeterminacy of model parameterizations; different parameterizations of these models yield accurate postdictions, but make disparate predictions.  Oftentimes, the parameter space of these models is characterized by extensive simulation with no probabilistic treatment of the solution space.  When both data and a reliable posterior inference method are available, probability can be assigned to each parameterization based on data likelihood and some prior assumptions.  Even in these cases, it is unclear how the structure of the posterior distribution relies on different properties of the data.  We present a new machine learning methodology for learning probabilistic degenerate solution spaces of models given statistical descriptions of the model behavior.  A degenerate solution network (DSN) learns a deep generative approximation to the maximally entropic distribution of model parameterizations that result in a model behavior of interest.

This tutorial explains how to use the dsn git repo to learn built-in or user-specified systems.  You should follow the [standard install](#standard-install) instructions if you only intend to learn DSNs for the built-in systems.  If you intend to write tensorflow code for your own system class, you should use the [dev install](#dev-install).  First we provide some background information covering the context of DSNs in the broader machine learning literature.  Then, we give an intutive explanation of what DSNs are using a classicly studied neural circuit.  We explain how to train deep generative density networks using code from the cunningham-lab github, as well as important diagnostics for checking approximation fidelity in DSNs.  We walk through a simple usage case learning the degenerate space of a 2-D linear system producing a gaussian band of oscillations.  Finally, we show how to use DSNs to learn solution spaces of dynamic computations using dynamic mean field theory for low-rank RNNs.

# Installation #
## Standard install<a name="standard-install"></a> ##
These installation instructions are for users interested in learning degenerate solution spaces of systems, which are already implemented in the [DSN systems library](systems.md).  Clone the git repo, and then go to the base directory and run the installer.
```bash
git clone https://github.com/cunningham-lab/dsn.git
cd dsn/
python setup.py install
```

## Dev install<a name="dev-install"></a> ##
If you intend to write some tensorflow code for your own system, then you want to use the development installation.  Clone the git repo, and then go to the base directory and run the development installer.
```bash
git clone https://github.com/cunningham-lab/dsn.git
cd dsn/
python setup.py develop
```

## Training a density network ##
To learn about training a density network to approximate a distribution, consult [this section](#) of the exponential family network (EFN) tutorial.

That section will have a lot of cool math like this:

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}](https://latex.codecogs.com/svg.latex?x%3D%5Cfrac%7B-b%5Cpm%5Csqrt%7Bb%5E2-4ac%7D%7D%7B2a%7D)
