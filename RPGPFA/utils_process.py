#%% Imports
from recognition_parametrised_gpfa import RPGPFA

import torch
import numpy as np
from torch import matmul
import matplotlib.pyplot as plt

import kernels

from flexible_multivariate_normal import FlexibleMultivariateNormal
from utils import diagonalize
from kernels import RBFKernel


def plot_loss(model, offset=0, **kwargs):
    plt.figure()
    plt.plot(model.loss_tot[offset:], c='k', lw=2, **kwargs)
    plt.ylabel('loss')
    plt.xlabel('Iterations')
    plt.title('- Free Energy')
    plt.tight_layout()


def plot_summary(model, plot_observation='all', plot_factors_id ='all',
                 latent_true=None, plot_regressed=False, plot_true=False, plot_variance=True, color_rgb=None):
    """
        Plot a summary of the latents discovered using RPGPFA

        Args:
            model (RPGPFA)
            plot_observation (str or list): observation if to plot 'all' / [x1, x2...]
            plot_factors_id (str or list): variational [-1], factors [0, 1, ...] or 'all'
            latent_true (str or list): if provided also plot the true latent
            plot_regressed (bool): regress latent true is provided, 'linear' or 'krr'
            plot_true (bool): plot true latent if provided
            plot_variance (bool): plot the marginal variance of discovered latent (if true latent not regressed)
            color_rgb (list): color plot of the latent

        Note:
            Some options are mutually exclusive
    """

    # Options
    plot_regressed = False if latent_true is None else plot_regressed
    plot_variance = False if plot_regressed else plot_variance
    plot_true = False if latent_true is None else plot_true

    # Dimension of the problem
    num_observation = model.num_observation
    len_observation = model.len_observation
    num_factors = model.num_factors

    dim_latent_fit = model.dim_latent

    if not latent_true is None:
        dim_latent_true = latent_true.shape[-1]



    plot_observation_id = plot_observation

    if plot_factors_id == 'all':
        plot_factors_id = np.arange(-1, num_factors)

    # Factor Names
    name_factors_all = ['E[q](Z)']
    for fc in range(num_factors):
        name_factors_all.append('E[f'+ str(fc) + '](Z)')
    name_factors = [name_factors_all[ii + 1] for ii in plot_factors_id]

    # Plot colors
    color_tot = get_xx_color(color_rgb, num_observation, len_observation)


    # Regress latent if necessary
    latent_fit_all = []
    latent_var_all = []
    for factors_id in plot_factors_id:
        if plot_regressed:
            latent_cur, _,  latent_true_plot, _ = regress_latent(model, factors_id, latent_true, regression=plot_regressed)
        else:
            latent_cur, latent_var_cur, _, _  = regress_latent(model, factors_id)
            latent_var_cur = latent_var_cur.diagonal(dim1=-1, dim2=-2)

        # Reshape Latents if necessary
        if plot_observation == 'all':
            folded_shape_latent_fit = latent_cur.shape
            unfolded_shape_latent_fit = (1, torch.tensor(folded_shape_latent_fit[:-1]).prod(), latent_cur.shape[-1])
            latent_cur = latent_cur.reshape(unfolded_shape_latent_fit)
            plot_observation_id = range(1)

            if plot_regressed:
                folded_shape_latent_true = latent_true.shape
                unfolded_shape_latent_true = (1, torch.tensor(folded_shape_latent_true[:-1]).prod(), latent_true.shape[-1])
                latent_true_plot = latent_true_plot.reshape(unfolded_shape_latent_true)

            else:
                latent_var_cur = latent_var_cur.reshape(unfolded_shape_latent_fit)

                if plot_true:
                    folded_shape_latent_true = latent_true.shape
                    unfolded_shape_latent_true = (1, torch.tensor(folded_shape_latent_true[:-1]).prod(), latent_true.shape[-1])
                    latent_true_plot = latent_true_plot.reshape(unfolded_shape_latent_true)

            color_tot = color_tot.reshape((*unfolded_shape_latent_fit[:-1], 3))

        else:
            if plot_true:
                latent_true_plot = latent_true

        latent_fit_all.append(latent_cur.detach().clone())
        if plot_variance:
            latent_var_all.append(latent_var_cur.detach().clone())

    # Plot Dimensions
    dim_latent_fit = latent_fit_all[0].shape[-1]
    heigh = len(plot_factors_id)
    width = dim_latent_fit

    # Plot all Fit and all factor for each dimensions
    for plot_obs_cur in plot_observation_id:
        plt.figure(figsize=(4 * width, 3 * heigh))
        for factors_id in range(heigh):
            for dim_cur in range(width):

                plt.subplot(heigh, width, 1 + factors_id * width + dim_cur)
                latent_plot = latent_fit_all[factors_id][plot_obs_cur, :, dim_cur]
                color_plot = color_tot[plot_obs_cur]

                if plot_regressed:
                    latent_true_cur = latent_true_plot[plot_obs_cur, :, dim_cur]
                    plt.plot(latent_true_cur, c='k', label = 'true', linestyle='-.')
                elif plot_variance:
                    latent_var_plot = latent_var_all[factors_id][plot_obs_cur, :, dim_cur]
                    latent_std_plot = torch.sqrt(latent_var_plot)
                    xx = range(len(latent_var_plot))
                    up = latent_plot + 2 * latent_std_plot
                    lo = latent_plot - 2 * latent_std_plot
                    plt.fill_between(xx, lo, up, color='k', alpha=.25)

                plt.plot(latent_plot, c='k', label='fit')
                plt.scatter(range(len(latent_plot)), latent_plot, c=color_plot)


                if dim_cur == 0:
                    plt.ylabel(name_factors[factors_id])
                    if factors_id == 0:
                        plt.legend()
                if factors_id == (heigh-1):
                    plt.xlabel('Time')
                elif factors_id == 0:
                    plt.title('Dim#' + str(dim_cur) + ' Obs#' + str(plot_obs_cur))
        plt.tight_layout()

    # Plot True Latent for each dimensions
    if plot_true:
        for plot_obs_cur in plot_observation_id:
            plt.figure(figsize=(4 * width, 3 * 1))
            for dim_cur in range(width):

                plt.subplot(1, width, 1 + dim_cur)
                latent_true_cur = latent_true_plot[plot_obs_cur, :, dim_cur]
                plt.plot(latent_true_cur, c='k', label = 'true')
                plt.scatter(range(len(latent_true_cur)), latent_true_cur, c=color_plot)

                if dim_cur == 0:
                    plt.ylabel('True')

                plt.xlabel('Time')
                plt.title('Dim#' + str(dim_cur) + ' Obs#' + str(plot_obs_cur))
                plt.tight_layout()

    # Plot Fit 2D or 3D view
    if dim_latent_fit == 3 or dim_latent_fit == 2:
        plt.figure(figsize=(5 * heigh, 5))
        for plot_obs_cur in plot_observation_id:
            for factors_id in range(heigh):

                if dim_latent_fit == 3:
                    ax = plt.subplot(1, heigh, 1 + factors_id, projection='3d')
                    latent_plot = latent_fit_all[factors_id][plot_obs_cur]
                    ax.scatter(latent_plot[:, 0], latent_plot[:, 1], latent_plot[:, 2], c=color_tot[plot_obs_cur], s=10)
                    plt.title(name_factors[factors_id])

                elif dim_latent_fit == 2:
                    plt.subplot(1, heigh, 1 + factors_id)
                    latent_plot = latent_fit_all[factors_id][plot_obs_cur]
                    plt.scatter(latent_plot[:, 0], latent_plot[:, 1], c=color_tot[plot_obs_cur], s=10)
                    plt.title(name_factors[factors_id])

    # Plot True 2D or 3D view
    if plot_true and (dim_latent_fit == 3 or dim_latent_fit == 2):

        plt.figure(figsize=(5 * 1, 5))
        for plot_obs_cur in plot_observation_id:
            if dim_latent_true == 3:
                ax = plt.subplot(1, 1, 1 , projection='3d')

                latent_true_cur = latent_true_plot[plot_obs_cur]
                ax.scatter(latent_true_cur[:, 0], latent_true_cur[:, 1], latent_true_cur[:, 2], c=color_tot[plot_obs_cur], s=10)
                plt.title('True')

            elif dim_latent_true == 2:
                latent_true_cur = latent_true_plot[plot_obs_cur]
                plt.scatter(latent_true_cur[:, 0], latent_true_cur[:, 1], c=color_tot[plot_obs_cur], s=10)
                plt.title('True')


def regress_latent(model, plot_factor_id=-1, latent_true=None, regression='krr'):

    # Grasp Variational or Factors Latent mean and variance
    if plot_factor_id < 0:
        latent_fit, latent_var = model.variational_marginals.mean_covariance()
        latent_fit = latent_fit.detach()
        latent_var = latent_var.detach()
    else:
        latent_fit, latent_var = model.factors.mean_covariance()
        latent_fit = latent_fit[plot_factor_id].detach()
        latent_var = latent_var[plot_factor_id].detach()

    # Regress Fit Latent to the True Latent
    if latent_true is None or not regression:
        regressor = None
    else:
        # Folded and Unfolded shape
        folded_shape_latent_fit = latent_fit.shape
        folded_shape_latent_true = latent_true.shape
        unfolded_shape_latent_fit = (torch.tensor(folded_shape_latent_fit[:-1]).prod(), latent_fit.shape[-1])
        unfolded_shape_latent_true = (torch.tensor(folded_shape_latent_true[:-1]).prod(), latent_true.shape[-1])

        # Unfold and 0 center Latents
        latent_true = latent_true.reshape(unfolded_shape_latent_true)
        latent_true = latent_true - latent_true.mean(dim=0)
        latent_fit = latent_fit.reshape(unfolded_shape_latent_fit)
        latent_fit = latent_fit - latent_fit.mean(dim=0)

        # Regress Latent - True Latent
        if regression == 'linear':
            latent_fit, regressor = regress_linear(latent_fit, latent_true)
        elif regression == 'krr':
            latent_fit, regressor = regress_krr(latent_fit, latent_true)
        else:
            raise NotImplementedError

        # Reshape True and Regressed latent
        latent_fit = latent_fit.reshape(folded_shape_latent_true)
        latent_true = latent_true.reshape(folded_shape_latent_true)

    return latent_fit, latent_var, latent_true, regressor


def get_xx_color(color_rgb, x1, x2):
    if color_rgb == None:

        c1 = torch.linspace(0, 1, x2).repeat(x1).unsqueeze(-1)
        c2 = torch.linspace(0.4, 0.4, x2).repeat(x1).unsqueeze(-1)
        c3 = torch.linspace(1, 0, x2).repeat(x1).unsqueeze(-1)

        m1 = torch.kron(torch.linspace(1, 1, x1).unsqueeze(-1), torch.ones(x2, 1))
        m2 = torch.kron(torch.linspace(0, 1, x1).unsqueeze(-1), torch.ones(x2, 1))
        m3 = torch.kron(torch.linspace(1, 1, x1).unsqueeze(-1), torch.ones(x2, 1))

        c1 = c1 * m1
        c2 = c2 * m2
        c3 = c3 * m3

        color_tot = torch.cat((c1, c2, c3), dim=-1)
        color_tot = color_tot.reshape(x1, x2, 3)

    else:
        color_tot = torch.tensor(color_rgb).unsqueeze(0).unsqueeze(0).repeat(x1, x2, 1)

    return color_tot


def sample_XYtrain(X, Y, train_pct):
    len_input = X.shape[0]
    len_train = int(len_input * train_pct)
    idx_train = np.random.choice(len_input, len_train)
    Xtrain = X[idx_train, :]
    Ytrain = Y[idx_train, :]

    return Xtrain, Ytrain

def regress_linear(X, Y, train_pct=0.5):

    Xtrain, Ytrain = sample_XYtrain(X, Y, train_pct)
    XXinv = torch.linalg.inv(1e-6 * torch.eye(Xtrain.shape[-1]) + torch.matmul(Xtrain.transpose(-1, -2), Xtrain))
    beta_hat = matmul(XXinv, matmul(Xtrain.transpose(-1, -2), Ytrain))

    def regressor(X0):
        return matmul(X0, beta_hat)

    Yhat = regressor(X)

    return Yhat, regressor


def regress_krr(X, Y, train_pct=0.5, kernel_param=None):

    if kernel_param is None:
        kernel_param = {'type': 'RBF', 'param': {'scale': torch.ones(1), 'lengthscale': torch.ones(1)}}
    if kernel_param['type'] == 'RBF':
        kernel = kernels.RBFKernel(**kernel_param['param'])
    if kernel_param['type'] == 'RQ':
        kernel = kernels.RQKernel(**kernel_param['param'])
    if kernel_param['type'] == 'POLY':
        kernel = kernels.POLYKernel(**kernel_param['param'])

    alpha = 1e-3
    Xtrain, Ytrain = sample_XYtrain(X, Y, train_pct)
    KXtrainXtrain = kernel.forward(Xtrain, Xtrain).squeeze(0)
    INN = torch.eye(KXtrainXtrain.shape[0])
    beta_hat = matmul(torch.linalg.inv(KXtrainXtrain + alpha * INN), Ytrain)

    def regressor(X0):
        KxXtrain = kernel.forward(X0, Xtrain).squeeze(0)
        return matmul(KxXtrain, beta_hat)

    Yhat = regressor(X)

    return Yhat, regressor

def plot_factors_prior(model, tt_index=None, factor_id=0):
    """
        Compare factor mixture and prior fitted from RPGPFA model

        Args:
            model (RPGPFA)
            tt_index (int): time index
    """
    # Data Type / Device
    dtype = model.dtype
    device = model.device

    # Z landscape
    dim_latent = model.dim_latent
    z_lanscape_tmp =torch.linspace(-15, 15, 50, dtype=dtype, device=device).unsqueeze(-1)
    z_landscape = z_lanscape_tmp.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)

    # Problem Dimensions
    len_observation = model.len_observation
    num_observation = model.num_observation
    num_factors = model.num_factors

    # Mean Function fitted
    mean_param, scale, lengthscale = model.prior_mean_param
    prior_mean_kernel = RBFKernel(scale, lengthscale)
    prior_mean = matmul(prior_mean_kernel(model.observation_locations, model.inducing_locations),
                        mean_param.unsqueeze(-1)).squeeze(-1).detach().clone().numpy()

    # Marginal Prior
    natural1prior, natural2prior = model._get_prior_marginals()
    covariance_prior = -0.5 / natural2prior
    mean_prior = covariance_prior * natural2prior
    prior = FlexibleMultivariateNormal(natural1prior.unsqueeze(-1), diagonalize(natural2prior.unsqueeze(-1)),
                                       init_natural=True, init_cholesky=False)
    prob_prior = torch.exp(prior.log_prob(z_lanscape_tmp.unsqueeze(1).unsqueeze(1))).permute(0, 2, 1).detach().numpy()


    # Factors Marginal Distributions
    factors_mean_marginal, factors_cova_marginal = model.factors.mean_covariance()
    factors_mean_marginal = factors_mean_marginal.unsqueeze(-1)
    factors_cova_marginal = factors_cova_marginal[..., range(dim_latent), range(dim_latent)].unsqueeze(-1).unsqueeze(-1)
    factors_marginal = FlexibleMultivariateNormal(factors_mean_marginal, factors_cova_marginal,
                                                  init_natural=False, init_cholesky=False)

    # Factor and Mixture densities
    prob_factors_all = torch.exp(factors_marginal.log_prob(z_landscape)).detach()
    prob_factors_mean = torch.exp(factors_marginal.log_prob(z_landscape)).detach().mean(1)

    prob_factors_all = prob_factors_all[:, factor_id]
    prob_factors_mean = prob_factors_mean[:, factor_id]

    # Time Index to Plot
    tt_index = 0 if tt_index is None else tt_index

    # Plot All recognition Factors, Priors and Mixture
    plt.figure(figsize=(4*prior_mean.shape[0], 4 * 2))
    for dd in range(prior_mean.shape[0]):
        plt.subplot(2, prior_mean.shape[0], dd +1)
        for nn in range(num_observation):
            # Current Factor mean
            fc = factors_mean_marginal[0, nn, :, dd, 0].detach().numpy()
            plt.plot(fc, color=[0.5, 0.5, 0.5])
        # Mixture Mean
        mc = factors_mean_marginal.mean(1)[0, :, dd, 0].detach().numpy()
        plt.plot(mc, color='k', label='Mixture Marginal Mean')
        # Prior Mean
        plt.plot(prior_mean[dd], color='m', label='Prior Marginal Mean')
        # Time of interest
        plt.scatter(tt_index, mc[tt_index], c='m', s=600)

        plt.title('Dim#' + str(dd) )
        plt.xlabel('Time [a.u]')
        if dd == 0:
            plt.ylabel('E(Z)')

    # Plot Marginal Distributions
    for dd in range(prior_mean.shape[0]):
        plt.subplot(2, prior_mean.shape[0], dd + 1 + prior_mean.shape[0])
        for nn in range(num_observation):
            # Factors
            pc = prob_factors_all[:, nn, tt_index, dd].detach().numpy()
            plt.plot(z_lanscape_tmp, pc, c=[0.5, 0.5, 0.5])
        # Mixture
        fc = prob_factors_mean[:, tt_index, dd].detach().numpy()
        plt.plot(z_lanscape_tmp, fc, c='k', label='Mixture Marginal PdF', linewidth=2)
        # Prior
        plt.plot(z_lanscape_tmp, prob_prior[:, tt_index, dd], linestyle='-.', color='m', label='Prior', linewidth=2)


        plt.tight_layout()
        plt.xlabel('Z[' + str(dd) + ']')
        if dd==0:
            plt.ylabel('p(z[t=' + str(tt_index) + '])')


    plt.tight_layout()
    plt.legend()


def plot_gradient_line(xx, cc, **kwargs):
    for tt in range(xx.shape[0]-1):
        plt.plot(xx[tt:tt+2, 0], xx[tt:tt+2,  1], c=cc[tt], **kwargs)
