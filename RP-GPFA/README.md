
# RP - GPFA

Up-to-date implementation of Recognition-Parametrised Gaussian Process Factor Analysis (RP-GPFA).
The model is summarized below, see
[Walker\*, Soulat\*, Yu, Sahani\* (2023)](https://arxiv.org/abs/2209.05661) for details.


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

Optimization uses a variational interior bound with additional constraint on the auxiliary factors.

## Repository

`recognition_parametrised_gpfa.py` instantiates and fits RP-GPFA.

`flexible_multivariate_normal.py` defines and handles Multivariate Normal Distributions using natural parameters.

We provide 3 Jupyter notebooks to illustrate RP-GPFA:

1) The structured bouncing ball experiment demonstrates the advantages of bypassing the specification a generative when
emphasis is put on latent identification. See:

    `./demo_textured_bouncing_ball.ipynb`

2) Lorenz attractor dynamics modulates high dimensional observation through on linearity and Poisson noise.

    `./demo_lorenz_attractor.ipynb`

3) 3D moving ellipsoid. RP-GPFA combines video and range sensor signal to infer the 2D position of an "agent" of interest.

    `./demo_textured_bouncing_ball.ipynb`


Dependencies are listed in `./rpgpfa.yml`

(The moving Ellipsoid dataset and stored demo results can be download [here](https://www.dropbox.com/sh/70yc801n3p64ke1/AAC3irVxD9p119N22J1qvqYYa?dl=0) ). 

