# Imports
import torch
import numpy as np

import _updates
import _initializations

from flexible_multivariate_normal import flexible_kl, kl
from utils import optimizer_wrapper, check_size, get_minibatches, print_loss

# TODO: Check the savings on a GPU ! Delete useless stuff in flexible MVN, utils, utils


class RPGPFA(_initializations.Mixin, _updates.Mixin):
    """
        Recognition Parametrised Gaussian Process Factor Analysis:
            We observe num_factors time-series measured over len_observation timesteps: X = {x_jt ~ dim_observation x 1}
            And seek to capture spatio-temporal structure in latent time series       : Z = {z_t  ~ dim_latent x 1}

        Args:
            observations (Tuple (len num_factors) or Tensor): shape num_observation x len_observation x dim_observation
            observation_locations (Tensor): shape len_observation x dim_locations. Location of the Observations
            inducing_locations (Tensor): shape len_observation x dim_locations. Location of inducing points
            fit_params (Dict): Fit parameters (dim_latent, inference method, lr, optimizer, etc...)

            Optional Args for initialisation:
                loss_tot (List), prior_mean_param (Tuple), prior_covariance_kernel (Kernel)
                inducing_points_param (Tuple) ,recognition_function (Net)

        Note:
            Quantities of interest from the fit include:
                factors (FlexibleMultivariateNormal)             : recognition distribution f(z_t| x_jt) ~ p(z_t | x_jt)
                variational_marginals (FlexibleMultivariateNormal): variational distributions q(z_t) ~ p(z_t | X)
    """

    def __init__(self, observations, observation_locations, inducing_locations=None, fit_params=None, loss_tot=None,
                 prior_mean_param=None, prior_covariance_kernel=None, inducing_points_param=None,
                 recognition_function=None):

        # Transform Observation in length 1 tuple if necessary
        observations = (observations,) if not (type(observations)) is tuple else observations

        # Device and data type
        self.dtype = observations[0].dtype
        self.device = observations[0].device
        str_device = 'GPU' if torch.cuda.is_available() else 'CPU'
        print('RP-GPFA on ' + str_device)

        # Check that observation dimensions are consistent across factors
        num_factors, num_observation, len_observation = check_size(observations)

        # Dimension of the problem
        self.num_factors = num_factors
        self.num_observation = num_observation
        self.len_observation = len_observation

        # Set observation/inducing locations
        self.observation_locations = observation_locations
        self.inducing_locations = observation_locations if inducing_locations is None else inducing_locations
        self.num_inducing_point = self.inducing_locations.shape[0]

        # Set all parameters with input
        self.fit_params = fit_params
        self.prior_mean_param = prior_mean_param
        self.prior_covariance_kernel = prior_covariance_kernel
        self.recognition_function = recognition_function
        self.inducing_points_param = inducing_points_param

        # Init all distributions
        self.prior = None
        self.factors = None
        self.log_weights = None
        self.factors_delta = None
        self.factors_param = None
        self.inducing_points = None
        self.variational_marginals = None

        # Initialise remaining parameters
        self._init_all(observations)

        # Set minibatches
        self.mini_batches = get_minibatches(
            self.fit_params['num_epoch'], self.len_observation, self.fit_params['minibatch_size'])
        self.epoch_batch = [0, 0]

        # Build all distributions
        self.update_all(observations, full_batch=True)

        # Init Loss
        self.loss_tot = [] if loss_tot is None else loss_tot

    def fit(self, observations):
        """ Fit RP-GPFA to observations """

        # Transform Observation in length 1 tuple if necessary
        observations = (observations,) if not (type(observations)) is tuple else observations

        # Grasp Fit parameters
        fit_params = self.fit_params

        # Epoch number
        num_epoch = fit_params['num_epoch']

        # Grasp minibatches
        mini_batches = self.mini_batches

        # Parameters to optimize
        prior_param = [*self.prior_mean_param, *self.prior_covariance_kernel.parameters()]
        inducing_point_param = [*self.inducing_points_param]
        factors_param = []
        for cur_factor in self.recognition_function:
            factors_param += cur_factor.parameters()

        # Gather Optimizers
        optimizer_prior = optimizer_wrapper(prior_param, fit_params['optimizer_prior'])
        optimizer_factors = optimizer_wrapper(factors_param, fit_params['optimizer_factors'])
        optimizer_inducing_points = optimizer_wrapper(inducing_point_param, fit_params['optimizer_inducing_points'])
        optimizers = [optimizer_prior, optimizer_factors, optimizer_inducing_points]

        # Training
        for epoch_id in range(num_epoch):

            # Current epoch losses
            loss_minibatch = []
            num_minibatch = len(mini_batches[epoch_id])

            for batch_id in range(num_minibatch):
                # Set current minibatch
                self.epoch_batch = [epoch_id, batch_id]

                # Update all distributions with current batch and parameters
                self.update_all(observations)

                # Step over all parameters
                loss_item = self._step_all(optimizers)

                # Current minibatch Loss
                loss_minibatch.append(loss_item)

            # Average epoch loss
            self.loss_tot.append(np.mean(loss_minibatch))
            print_loss(self.loss_tot[-1], epoch_id + 1, num_epoch, pct=self.fit_params['pct'])

        return self.loss_tot

    # %% Loss / Free Energy estimation
    def _get_loss(self):
        """ Estimate loss defined as an upper bound of the negative Free Energy """

        # Dimension of the problem
        num_observation = self.num_observation
        len_observation = self.fit_params['minibatch_size']

        # KL[variational, prior] for Inducing points ~ num_observations x dim_latent
        KLqUpU = flexible_kl(self.inducing_points, self.prior, repeat1=[0, 1], repeat2=[1])

        # log Gamma - KL[variational, factors_delta]  ~ num_factors x num_observations x len_observations
        interior_vlb = self._get_free_energy_lower_bound()

        # - loss
        free_energy = (- KLqUpU.sum() + interior_vlb.sum()) / (num_observation * len_observation)

        return - free_energy

    def _get_free_energy_lower_bound(self):
        """ Variational Lower Bound Of the Free Energy """

        # Dimension of the problem
        num_observation = self.num_observation

        # Update the ratio distributions
        log_weights = self.log_weights
        factors_delta = self.factors_delta

        # Normalisers ~ num_factors x num_observations x len_observations
        log_gamma = log_weights.diagonal(dim1=1, dim2=2).permute(0, 2, 1)

        # Natural Parameter from the marginal
        variational_natural1 = self.variational_marginals.natural1.unsqueeze(0)
        variational_natural2 = self.variational_marginals.natural2.unsqueeze(0)
        variational_log_normalizer = self.variational_marginals.log_normalizer.unsqueeze(0)
        variational_suff_stat = [self.variational_marginals.suff_stat_mean[0].unsqueeze(0),
                                 self.variational_marginals.suff_stat_mean[1].unsqueeze(0)]

        # Grasp only the m = n distribution for KL estimation
        diag_id = range(num_observation)
        diag_delta_natural1 = factors_delta.natural1[:, diag_id, diag_id]
        diag_delta_natural2 = factors_delta.natural2[:, diag_id, diag_id]
        diag_delta_factors_log_normalizer = factors_delta.log_normalizer[:, diag_id, diag_id]

        # KL(q || fhat) ~ num_factors x num_observations x len_observations
        KLqfhat = kl((variational_natural1, variational_natural2),
                     (diag_delta_natural1, diag_delta_natural2),
                     variational_log_normalizer, diag_delta_factors_log_normalizer,
                     variational_suff_stat)

        return log_gamma - KLqfhat

    # %% Initializations, Updates and Steps
    def _init_all(self, observations):
        """ Init all parameters (see _initializations.Mixin) """

        # Fit parameters
        if self.fit_params is None:
            self.fit_params = {}
        self._init_fit_params()
        self.dim_latent = self.fit_params['dim_latent']

        # Covariance function of the prior Gaussian Process
        if self.prior_covariance_kernel is None:
            self._init_kernel()

        # Mean Parametrization of the prior Gaussian Process
        if self.prior_mean_param is None:  # TODO : integrate this
            self._init_prior_mean_param()

        # Recognition neural networks
        if self.recognition_function is None:
            self._init_recognition(observations)

        # Inducing Points variational distribution parameters
        if self.inducing_points_param is None:
            self._init_inducing_points()

    def update_all(self, observations, full_batch=False):
        """Update all distributions with current parameter values (see _updates.Mixin)"""

        # Temporarily use full batch updates
        if full_batch:
            buffer = (self.epoch_batch, self.mini_batches)
            self.epoch_batch = [0, 0]
            self.mini_batches = [[list(np.arange(self.len_observation))]]

        # Update Variational Distributions
        self._update_prior()
        self._update_inducing_points()
        self._update_variational_marginals()

        # Update factors and auxiliary factors
        self._update_factors_tilde()
        self._update_factors(observations)
        self._update_factors_delta()

        # Reset current minibatch
        if full_batch:
            self.epoch_batch = buffer[0]
            self.mini_batches = buffer[1]

    def _step_all(self, optimizers):
        """ Get Loss and step over all optimized parameters """

        for opt in optimizers:
            opt.zero_grad()

        loss = self._get_loss()
        loss.backward()

        for opt in optimizers:
            opt.step()

        return loss.item()