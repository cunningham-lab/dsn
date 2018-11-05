---
layout: default
---
# Introduction #
Theoretical neuroscientists have complemented experimental investigations in our field by designing and testing mathematical models of neural activity [1]. Free of the burdens of experimental neuroscience, theorists do this with rapid turnover, frequently rejecting and sometimes promoting new models, which make accurate post-dictions of established experiments.  Theoretical neuroscience is most powerful when it presents new models that make consequential and experimentally testable predictions [cite examples].  This scientific paradigm suffers from the general issue of indeterminacy; different parameterizations of these models make accurate post-dictions, but yield disparate predictions.  In the absence of data, an observation model, and a high-fidelity posterior inference method, we lack the ability to probabilistically characterize this solution space, making it difficult to assess which predictions are appropriate to make under a model and the system’s known behavior.  

Using recent advancements in deep generative modeling, degenerate solution networks (DSNs) learn the full distribution of model parameterizations that yield a system behavior of interest. Maybe you want to use a DSN for X.  Maybe you want to use a DSN for Y and then Z.  The dsn repo is intended to facilitate all of those needs.  This tutorial explains how to use the repo to learn built-in or user-designed systems.

# Installation #
## Standard ##
These installation instructions are for users interested in learning degenerate solution spaces of systems, which are already implemented in the [DSN systems library](systems.md).
```bash
echo "hello world"
```

## Dev installation ##
If you intend to write some tensorflow code for your own system, then you want to use the development installation.  
#### Syntax hilighting ####
```python
import sys
learn_init = True;
if os.path.exists(initfname):

    resfile = np.load(resfname);
    assert(resfile['converged']);

else:
    print('here we go with the NF');
```