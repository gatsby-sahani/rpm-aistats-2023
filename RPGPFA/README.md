
# RP - GPFA

This repository contains the compagnon code for the Recognition-Parametrised Gaussian Process Factor Analysis (RP-GPFA) experiments.

## Model

RP-GPFA models continuous multi-factorial temporal dependencies. We consider $J$ observed time-series measured over $T$ timesteps:

$$\mathcal{X} = \{ \mathsf{x}_{jt} : j = 1\dots J, t=1\dots T \}$$

We seek to capture both spatial and temporal structure in a set of $K$-dimensional underlying latent time-series:

$$\mathcal{Z}=\{\mathsf{z}_t:t=1 \dots T\}$$

such that the observations are  conditionally independent across series and across time. The full joint writes:

$$ P_{\theta, \mathbb{X}^N}(\mathcal{X}, \mathcal{Z}) = \prod_{j=1}^J \prod_{t=1}^T \left( p_{0,jt}(\mathsf{x}_{jt}) \frac{f_{\theta j}(\mathsf{z}_{t} | \mathsf{x}_{jt})}{F_{\theta j}(\mathsf{z}_{t})} \right) p_{\theta z}(\mathcal{Z})$$


Each recognition factor is parametrised by a neural network $\theta_j$ that outputs the natural parameters of a multivariate normal distribution given input $\mathsf{x}_{jt}^{(n)}$ and we recall that

$$F_{\theta j}(\mathsf{z}_{t}) = \frac1N \sum_{n=1}^N f_{\theta j}(\mathsf{z}_{t} | \mathsf{x}_{jt}^{(n)})$$

Finally, the prior $p_{\theta z}(\mathcal{Z})$ comprises independent Gaussian Process priors over each latent dimension.


## Repository

We provide the code used in our paper as well as a jupyter notebook to test:

1) the RP-GPFA on structured bouncing ball experiment. Inference uses the Interior Varional Bound method.
   `./demo_rp_gpfa_textured_ball.ipynb`

2) the RP-GPFA on a bouncing ball experiment with Poisson Emission noise. Inference uses the 2nd Order Approximation method.
    `./demo_rp_gpfa_poisson_ball.ipynb`

Dependencies are listed in `./rpgpfa.yml`


`UnstructuredRecognition` instantiates and fits RP-GPFA.

`FlexibleMultivariate` defines and handles Multivariate Normal Distributions using natural parameters.
