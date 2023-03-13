{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true,
    "pycharm": {
     "name": "#%% md\n"
    }
   },
   "source": [
    "# RP - GPFA\n",
    "\n",
    "Up-to-date implementation of Recognition-Parametrised Gaussian Process Factor Analysis (RP-GPFA).\n",
    "The model is summarized below, see\n",
    "[Walker\\*, Soulat\\*, Yu, Sahani\\* (2023)](https://arxiv.org/abs/2209.05661) for details.\n",
    "\n",
    "\n",
    "## Model\n",
    "\n",
    "RP-GPFA models continuous multi-factorial temporal dependencies. We consider $J$ observed time-series measured over $T$ timesteps:\n",
    "\n",
    "$$\\mathcal{X} = \\{ \\mathsf{x}_{jt} : j = 1\\dots J, t=1\\dots T \\}$$\n",
    "\n",
    "We seek to capture both spatial and temporal structure in a set of $K$-dimensional underlying latent time-series:\n",
    "\n",
    "$$\\mathcal{Z}=\\{\\mathsf{z}_t:t=1 \\dots T\\}$$\n",
    "\n",
    "such that the observations are  conditionally independent across series and across time. The full joint writes:\n",
    "\n",
    "$$ P_{\\theta, \\mathbb{X}^N}(\\mathcal{X}, \\mathcal{Z}) = \\prod_{j=1}^J \\prod_{t=1}^T \\left( p_{0,jt}(\\mathsf{x}_{jt}) \\frac{f_{\\theta j}(\\mathsf{z}_{t} | \\mathsf{x}_{jt})}{F_{\\theta j}(\\mathsf{z}_{t})} \\right) p_{\\theta z}(\\mathcal{Z})$$\n",
    "\n",
    "\n",
    "Each recognition factor is parametrised by a neural network $\\theta_j$ that outputs the natural parameters of a multivariate normal distribution given input $\\mathsf{x}_{jt}^{(n)}$ and we recall that\n",
    "\n",
    "$$F_{\\theta j}(\\mathsf{z}_{t}) = \\frac1N \\sum_{n=1}^N f_{\\theta j}(\\mathsf{z}_{t} | \\mathsf{x}_{jt}^{(n)})$$\n",
    "\n",
    "Finally, the prior $p_{\\theta z}(\\mathcal{Z})$ comprises independent Gaussian Process priors over each latent dimension.\n",
    "\n",
    "Optimization uses a variational interior bound with additional constraint on the auxiliary factors.\n",
    "\n",
    "## Repository\n",
    "\n",
    "`recognition_parametrised_gpfa.py` instantiates and fits RP-GPFA.\n",
    "\n",
    "`flexible_multivariate_normal.py` defines and handles Multivariate Normal Distributions using natural parameters.\n",
    "\n",
    "We provide 3 Jupyter notebooks to illustrate RP-GPFA:\n",
    "\n",
    "1) The structured bouncing ball experiment demonstrates the advantages of bypassing the specification a generative when\n",
    "emphasis is put on latent identification. See:\n",
    "\n",
    "    `./demo_textured_bouncing_ball.ipynb`\n",
    "\n",
    "2) Lorenz attractor dynamics modulates high dimensional observation through on linearity and Poisson noise.\n",
    "\n",
    "    `./demo_lorenz_attractor.ipynb`\n",
    "\n",
    "3) 3D moving ellipsoid. RP-GPFA combines video and range sensor signal to infer the 2D position of an \"agent\" of interest.\n",
    "\n",
    "    `./demo_textured_bouncing_ball.ipynb`\n",
    "\n",
    "\n",
    "Dependencies are listed in `./rpgpfa.yml`\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "source": [
    "print(0)"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   },
   "execution_count": 16,
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n"
     ]
    }
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}