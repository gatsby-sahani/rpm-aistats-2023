# RPM (discrete)

This repository contains the compagnon code for the discrete Recognition-Parametrised Model (RPM).

RPM is a new approach to probabilistic unsupervised learning for joint distributions over observed and latent variables.
It uses the key assumption that observations

$$\mathcal{X} = \{\mathsf{x}_{j} : j = 1\dots J\}$$

are conditionally independent given the latents

$$\mathcal{Z} = \{\mathsf{z}_{l} : l = 1\dots L\}$$.

Given empirical measures $p_{0j}$, prior $p_{\theta z}$ and recognition factors $f_{\theta j}$, it writes

$$ P_{\theta, \mathbb{X}^N}(\mathcal{X}, \mathcal{Z}) = \prod_{j=1}^J \left( p_{0,j}(\mathsf{x}_{j}) \frac{f_{\theta j}(\mathcal{z} | \mathsf{x}_{j})}{f_{\theta j}(\mathcal{z})} \right) p_{\theta z}(\mathcal{Z})$$


## Repository

We provide the code used in our paper as well as jupyter notebooks to test:

1) RPM-Peer Supervision
`./demo_rpm_peer_supervision.ipynb`

2) RPM-Hidden Markov Model
`./demo_rpm_hmm_maze.ipynb`

Dependencies are listed in ./rpm.yml
