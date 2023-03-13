# Imports
import torch
from torch import matmul

import kernels
from utils import diagonalize, minibatch_tupple
from flexible_multivariate_normal import FlexibleMultivariateNormal, NNFlexibleMultivariateNormal, \
    vector_to_tril


class Mixin:
    """
        Mixin class containing necessary methods for updating RPGPFA model distributions
        For clarity, we denote: J:num_factors,  N:num_observations, T:len_observations, K:dim_latent, M:num_inducing
    """

    def _update_prior(self):
        """ Build GP-Prior at the inducing locations """

        # Mean function at inducing locations
        mean_param, scale, lengthscale = self.prior_mean_param
        prior_mean_kernel = kernels.RBFKernel(scale, lengthscale, copy_scale=False, copy_lenghscale=False)
        mean = matmul(prior_mean_kernel.forward(self.inducing_locations, self.inducing_locations),
                      mean_param.unsqueeze(-1)).squeeze(-1)

        # Covariance function at inducing locations
        covariance = self.prior_covariance_kernel(self.inducing_locations, self.inducing_locations)

        # Build Prior
        self.prior = FlexibleMultivariateNormal(mean, covariance, init_natural=False, init_cholesky=False)

    def _update_inducing_points(self):
        """ Build Inducing points Variational Distributions from natural parameters """

        # Natural Parameters
        natural1, natural2_chol_vec = self.inducing_points_param
        natural2_chol = vector_to_tril(natural2_chol_vec)

        self.inducing_points = FlexibleMultivariateNormal(natural1, natural2_chol, init_natural=True,
                                                          init_cholesky=True, store_suff_stat_mean=True)

    def _update_variational_marginals(self):
        """ Update Latent Variational Distribution from Inducing Point Distribution """

        # Inducing Points (tau ~ M) and Observations (t ~ T) depending on the current minibatch
        inducing_locations, observation_locations = self.get_locations()

        # Kernel Posterior helpers
        K_t_t, K_tau_tau, _, K_t_tau, K_t_tau_K_tau_tau_inv = \
            self.prior_covariance_kernel.posteriors(inducing_locations, observation_locations)

        # Cov_k(t, tau) inv( Cov_k(tau, tau) ) unsqueezed to ~ 1 x K x T x 1 x M
        K_t_tau_K_tau_tau_inv = K_t_tau_K_tau_tau_inv.unsqueeze(0).unsqueeze(-2)

        # Mean of the inducing Points ~ N x K x M
        inducing_mean = self.inducing_points.suff_stat_mean[0]

        # Covariance of the inducing points ~ N x K x M x M
        inducing_covariance = self.inducing_points.suff_stat_mean[1] \
                              - matmul(inducing_mean.unsqueeze(-1), inducing_mean.unsqueeze(-2))

        # inducing_covariance - Cov_k(tau, tau) unsqueezed to ~ N x K x 1 x M x M
        delta_K = (inducing_covariance - K_tau_tau.unsqueeze(0)).unsqueeze(-3)

        # Variational Marginals Mean Reshaped and Permuted to ~ N x T x K
        mean_param, scale, lengthscale = self.prior_mean_param
        prior_mean_kernel = kernels.RBFKernel(scale, lengthscale, copy_scale=False, copy_lenghscale=False)

        kernel_t_tau = prior_mean_kernel.forward(observation_locations, inducing_locations)
        kernel_tau_tau = prior_mean_kernel.forward(inducing_locations, inducing_locations)

        prior_mean_t = matmul(kernel_t_tau, mean_param.unsqueeze(-1)).squeeze(-1).unsqueeze(0)  # ~ 1 x K x T
        prior_mean_tau = matmul(kernel_tau_tau, mean_param.unsqueeze(-1)).squeeze(-1).unsqueeze(0)  # ~ 1 x K x M

        # Variational Marginals Mean Reshaped and Permuted to ~ N x T x K
        marginal_mean = (
                prior_mean_t + matmul(
            K_t_tau_K_tau_tau_inv, (inducing_mean - prior_mean_tau).unsqueeze(-2).unsqueeze(-1)
        ).squeeze(-1).squeeze(-1)
        ).permute(0, 2, 1)

        # Variational Marginals Covariance Reshaped and Permuted to ~ N x T x K (note that dimensions are independent)
        marginal_covariance_diag = (K_t_t.unsqueeze(0) + matmul(matmul(K_t_tau_K_tau_tau_inv, delta_K),
                                                                K_t_tau_K_tau_tau_inv.transpose(-1, -2)
                                                                ).squeeze(-1).squeeze(-1)).permute(0, 2, 1)

        # Square Root and Diagonalize the marginal Covariance ~ N x T x K x K (Alternatively, use 1D MVN)
        marginal_covariance_chol = diagonalize(torch.sqrt(marginal_covariance_diag))

        # Marginals distributions
        self.variational_marginals = FlexibleMultivariateNormal(marginal_mean, marginal_covariance_chol,
                                                                init_natural=False, init_cholesky=True,
                                                                store_suff_stat_mean=True)

    def _get_prior_marginals(self):
        """ Prior natural parameters associated with marginals at observation locations """

        # Dimensions of the problem (and batches)
        inducing_locations, observation_locations = self.get_locations()
        len_observation = observation_locations.shape[0]

        # Prior Covariance Function
        prior_covariance = self.prior_covariance_kernel(observation_locations, observation_locations)
        prior_marginal_covariance = prior_covariance[..., range(len_observation), range(len_observation)]

        # Prior mean
        mean_param, scale, lengthscale = self.prior_mean_param
        prior_mean_kernel = kernels.RBFKernel(scale, lengthscale, copy_scale=False, copy_lenghscale=False)
        kernel_t_tau = prior_mean_kernel.forward(observation_locations, inducing_locations)
        prior_mean = matmul(kernel_t_tau, mean_param.unsqueeze(-1)).squeeze(-1)

        # Deduce marginal natural parameters
        natural2_prior = - 0.5 / (prior_marginal_covariance + 1e-6)
        natural1_prior = - 2 * natural2_prior * prior_mean

        return natural1_prior, natural2_prior

    def _update_factors(self, observations):
        """  Build factor distributions from recognition function output """

        # Setting and dimensions
        dtype = self.dtype
        device = self.device
        dim_latent = self.dim_latent
        num_factors = self.num_factors

        # Grasp current minibatch of observation
        epoch_id, batch_id = self.epoch_batch
        mini_batch_cur = self.mini_batches[epoch_id][batch_id]
        observations_cur = minibatch_tupple(observations, dim=1, idx=mini_batch_cur, device=self.device)

        # Use this instead of self.x in case of minibatch
        num_observation, len_observation = observations_cur[0].shape[:2]

        # Prior Distribution marginals
        natural1_prior_tmp, natural2_prior_tmp = self._get_prior_marginals()
        natural1_prior = natural1_prior_tmp.permute(1, 0) \
            .unsqueeze(0).unsqueeze(0).repeat(num_factors, num_observation, 1, 1)
        natural2_prior = diagonalize(natural2_prior_tmp.permute(1, 0)) \
            .unsqueeze(0).unsqueeze(0).repeat(num_factors, num_observation, 1, 1, 1)

        # Init Natural Parameters ~ J x N x T x K
        natural1 = torch.zeros(num_factors, num_observation, len_observation, dim_latent,
                               dtype=dtype, device=device, requires_grad=False)
        natural2_chol = torch.zeros(num_factors, num_observation, len_observation, dim_latent, dim_latent,
                                    dtype=dtype, device=device, requires_grad=False)

        # Init Factors param (used to backpropagate natural gradients)
        factors_param = []

        for cur_factor in range(self.num_factors):
            # Grasp current recognition network
            cur_recognition = self.recognition_function[cur_factor]

            # Reshape For conv2d
            unfolded_shape = (num_observation * len_observation, *list(observations[cur_factor].shape[2:]))
            refolded_shape = (num_observation, len_observation, int(dim_latent + dim_latent * (dim_latent + 1) / 2))
            cur_observation = observations_cur[cur_factor].view(unfolded_shape).unsqueeze(1)

            # Recognition Function (+reshaping)
            cur_factors_param = cur_recognition(cur_observation).view(refolded_shape)

            # Store
            factors_param.append(cur_factors_param)

            # 1st Natural Parameter
            natural1[cur_factor] = cur_factors_param[..., :dim_latent]

            # Cholesky Decomposition of the (-) second natural parameter
            natural2_chol[cur_factor] = vector_to_tril(cur_factors_param[..., dim_latent:])

        # Build factor distributions
        natural1 = natural1_prior + natural1
        natural2 = natural2_prior + matmul(-natural2_chol, natural2_chol.transpose(-1, -2))

        self.factors_param = factors_param
        self.factors = FlexibleMultivariateNormal(natural1, natural2,
                                                  init_natural=True, init_cholesky=False,
                                                  store_natural_chol=True, store_suff_stat_mean=True)

    def _update_factors_tilde(self):
        """  Build (constrained auxiliary factors) """

        # Prior Distribution marginals
        num_factors = self.num_factors
        num_observations = self.num_observation
        natural1_prior_tmp, natural2_prior_tmp = self._get_prior_marginals()
        natural1_prior = natural1_prior_tmp.permute(1, 0) \
            .unsqueeze(0).unsqueeze(0).repeat(num_factors, num_observations, 1, 1)
        natural2_prior = diagonalize(natural2_prior_tmp.permute(1, 0)) \
            .unsqueeze(0).unsqueeze(0).repeat(num_factors, num_observations, 1, 1, 1)

        # Variational Distribution marginals
        natural1_variational_marginals = self.variational_marginals.natural1.unsqueeze(0)
        natural2_variational_marginals = self.variational_marginals.natural2.unsqueeze(0)

        # Auxiliary Factors
        natural1_tilde = natural1_prior - natural1_variational_marginals
        natural2_tilde = natural2_prior - natural2_variational_marginals

        # Store pseudo natural parameters
        self.factors_tilde = NNFlexibleMultivariateNormal(natural1_tilde, natural2_tilde)

    def _update_factors_delta(self):
        """Build all the ration distributions (factors - factors_tilde)"""

        # Natural Parameters of the factors ~ J x 1 x N x T x K (x K)
        factors_natural1 = self.factors.natural1.unsqueeze(1)
        factors_natural2 = self.factors.natural2.unsqueeze(1)
        factors_log_normaliser = self.factors.log_normalizer.unsqueeze(1)

        # Pseudo Natural Parameters of the auxiliary factors ~ J x N x 1 x T x K (x K)
        factors_tilde_natural1 = self.factors_tilde.natural1.unsqueeze(2)
        factors_tilde_natural2 = self.factors_tilde.natural2.unsqueeze(2)

        # eta_m - eta_tilde_n ~ J x N x N x T x K (x K)
        delta_natural1 = factors_natural1 - factors_tilde_natural1
        delta_natural2 = factors_natural2 - factors_tilde_natural2

        self.factors_delta = FlexibleMultivariateNormal(delta_natural1, delta_natural2,
                                                        init_natural=True, init_cholesky=False,
                                                        store_suff_stat_mean=True)

        # Ratio of log-nomaliser differences ~ J x N x N x T
        delta_log_normalizer = self.factors_delta.log_normalizer - factors_log_normaliser

        # In the ergodic cae, the sum is over T and N
        if self.fit_params['ergodic']:
            log_weights = delta_log_normalizer - torch.logsumexp(delta_log_normalizer, dim=(2, 3), keepdim=True)
        else:
            log_weights = delta_log_normalizer - torch.logsumexp(delta_log_normalizer, dim=2, keepdim=True)

        self.log_weights = log_weights

    def get_locations(self):
        """Use the minibatch variables to get current observation / inducing locations"""

        # Epoch / Mini Batch indices
        epoch_idx, batch_idx = self.epoch_batch

        # Inducing Points locations (M)
        inducing_locations = self.inducing_locations

        # Observations Locations (T or Mini Batch Length)
        mini_batch_idx = self.mini_batches[epoch_idx][batch_idx]
        observation_locations = self.observation_locations[mini_batch_idx]

        return inducing_locations, observation_locations