---
layout: default
---

<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML' async></script>


<div class="topnav">
  <a class="active" href="/index">Home</a>
  <a href="/systems">DSN Systems Library</a>
</div>

Theoretical neuroscientists design and test mathematical models of neural activity, assessing a model's quality by its ability to replicate experimental results.  A general challenge has been addressing the indeterminacy of model parameterizations; different parameterizations of these models accurately reflect established experiments, yet make incompatible predictions about other aspects of neural activity.  Degenerate solution networks (DSNs) learn the full (i.e. maximum entropy) distribution of generative model parameterizations that yield a behavior of interest.  This is a tool designed for theorists, enabling a new class of model analyses relying on the full distribution of generative parameters that result in some statistically specified activity.

This tutorial explains how to use the dsn git repo to learn built-in or user-specified systems. \\

# Installation #

You should follow the [standard install](#standard-install) instructions if you only intend to learn DSNs for the built-in systems.  If you intend to write tensorflow code for your own system class, you should use the [dev install](#dev-install).

## Standard install<a name="standard-install"></a> ##
These installation instructions are for users interested in learning degenerate solution spaces of systems, which are already implemented in the [DSN systems library](systems.md).  Clone the git repo, go to the base directory, and run the installer.
```bash
git clone https://github.com/cunningham-lab/dsn.git
cd dsn/
python setup.py install
```

## Dev install<a name="dev-install"></a> ##
If you intend to [write some tensorflow code for your own system](#)(<- TODO add link for this), then you want to use the development installation.  Clone the git repo, and then go to the base directory and run the development installer.
```bash
git clone https://github.com/cunningham-lab/dsn.git
cd dsn/
python setup.py develop
```

# Degenerate Solution Networks (DSNs) #
Consider model parameterization $$z$$ and data $$x$$ generated from some generative model of interest with known sampling procedure and likelihood $$p(x \mid z)$$, which may or may not be known.  Returning to the STG example, we have a known sampling procedure for simulating activity given a circuit parameterization, yet lack an explicit likelihood function for the generated neural activity due to the complex nonlinear dynamics.  In deep generative models, a simple random variable $$w \sim p_0$$ is mapped deterministically via a function $$f_\theta$$ parameterized by a neural network to the support of the distribution of interest where $$z = f_{\theta}(w)$$.

![](dsn.png)

Given a generative model $$p(x \mid z)$$ and some behavior of interest $$\mathcal{B}: E_{z \sim q_\theta}\left[ E_{x\sim p(x \mid z)}\left[T(x)\right] \right] = \mu$$, DSNs are trained by optimizing the deep generative parameters $$\theta$$ to find the optimal approximation $$q_{\theta}^*$$ within the deep generative variational family $$Q$$ to $$p(z \mid \mathcal{B})$$. This procedure is loosely equivalent to variational inference (VI) using a deep generative variational family with respect to the likelihood of the mean sufficient statistics rather than the data itself (Loaiza-Ganem et al. 2017, Bittner & Cunningham 2018)  In most settings (especially those relevant to theoretical neuroscience) the likelihood of the behavior with respect to the model parameters $$p(T(x) \mid z)$$ is unknown or intractable, requiring an alternative to stochastic gradient variational bayes (Kingma & Welling 2013) or black box variational inference (Ranganath et al. 2014). Instead, DSNs are optimized with the following objective for a given generative model and statistical constraints on its produced activity:

\begin{equation}
q_\theta^*(z) = \mathop{\arg\,\max}\limits_{q_\theta \in Q} H(q_\theta(z))
\end{equation}
\begin{equation}
 \text{s.t.  } E_{z \sim q_\theta}\left[ E_{x\sim p(x \mid z)}\left[T(x)\right] \right] = \mu
\end{equation}

# Example 1: An oscillating 2-D linear system. #

To provide intuition for DSNs, we learn the degenerate solution space of a 2-D linear system producing a gaussian distribution of oscillations.  For a linear system

\begin{equation}
\dot{x} = Ax, A = \begin{bmatrix} a_1 & a_2 \\\\ a_3 & a_4 \end{bmatrix}
\end{equation}

we learn the space of real entries of A that results in a band of oscillations: such that 
\begin{equation}
\text{real}(\lambda) \sim \mathcal{N}(0, 1)
\end{equation}
\begin{equation}
\text{imag}(\lambda) \sim \mathcal{N}(4\pi, 1)
\end{equation}
To do this, we train a DSN for this system, such that the first- and second-moment constraints are satisfied.  Even though we can compute (expect value of behavior) via the eigenvalues of A in closed form, we can not derive the distribution qthetastar, since the backward mapping from mu to eta of the exponential family is unknown.  Instead, we can train a DSN to learn the multimodal degenerate linear system parameterization.

## Training the DSN ##
mix of text and code describing training procedure.

[show dsn of 2-d linear system]
Other model-behavior combinations (see V1 example) will have even more complexity, further motivating DSNs.

## Augmented lagrangian optimization ##

# Example 2: 4-neuron V1 modeling #

## this is how we do it ##

# References #
 Sean Bittner and John Cunningham. *Learning exponential families.* (In review, AI Stats), ?(?):?-?, 2018.

Brian DePasquale, Christopher J Cueva, Kanaka Rajan, LF Abbott, et al. *full-force: A target-based method for training recurrent networks.* PloS one, 13(2):e0191527, 2018.

Diederik P Kingma and Max Welling. *Auto-encoding variational bayes.* arXiv preprint arXiv:1312.6114, 2013.

Gabriel Loaiza-Ganem, Yuanjun Gao, and John P Cunningham. *Maximum entropy flow networks.* arXiv preprint arXiv:1701.03504, 2017.

James Martens and Ilya Sutskever. *Learning recurrent neural networks with hessian-free optimization.* In Proceedings of the 28th International Conference on Machine Learning (ICML-11), pages 1033-1040. Citeseer, 2011.

Rajesh Ranganath, Sean Gerrish, and David Blei. *Black box variational inference.* In Artificial Intelligence and Statistics, pages 814-822, 2014.

Danilo Jimenez Rezende and Shakir Mohamed. *Variational inference with normalizing flows.* arXiv preprint arXiv:1505.05770, 2015.

