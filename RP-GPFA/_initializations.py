# Imports
import torch
import numpy as np
import torch.nn.functional as F
from torch.linalg import  cholesky

import kernels
from networks import Net
from utils import diagonalize
from flexible_multivariate_normal import tril_to_vector


class Mixin:
    """
        Mixin class containing necessary methods for initializing RPGPFA model
    """

    def _init_fit_params(self):
        """ Default Fit parameters """

        # Recognition Network(s) Hidden Dimension
        if not ('dim_hidden' in self.fit_params.keys()):
            # Init with same network for each factors
            self.fit_params['dim_hidden'] = tuple([[50, 50] for _ in range(self.num_factors)])
        else:
            # Check that for each factor hidden layer dimensions are provided
            assert len(self.fit_params['dim_hidden']) == self.num_factors

        # Recognition Network(s) Non Linearity
        if not ('nonlinearity' in self.fit_params.keys()):
            # Init with same network for each factors
            self.fit_params['nonlinearity'] = tuple([F.relu for _ in range(self.num_factors)])
        else:
            # Check that for each factor non linearity is provided
            assert len(self.fit_params['nonlinearity']) == self.num_factors

        # Recognition Network(s) Type (perceptron / convolutional)
        if not ('nn_type' in self.fit_params.keys()):
            # Init with similar network for each factors
            self.fit_params['nn_type'] = tuple(['perceptron' for _ in range(self.num_factors)])
        else:
            # Check that for each factor type are provided
            assert len(self.fit_params['nn_type']) == self.num_factors

        # Latent dimensions
        if not ('dim_latent' in self.fit_params.keys()):
            self.fit_params['dim_latent'] = 1

        # Constraint imposed on recognition factors
        if not ('constraint_factors' in self.fit_params.keys()):
            self.fit_params['constraint_factors'] = 'fixed'

        # Prior Gaussian Process Covariance Kernel
        if not ('gp_kernel' in self.fit_params.keys()):
            self.fit_params['gp_kernel'] = 'RBF'

        # Iterations
        if not ('num_epoch' in self.fit_params.keys()):
            self.fit_params['num_epoch'] = 500

        # Default is Full batch
        if not ('minibatch_size' in self.fit_params.keys()):
            self.fit_params['minibatch_size'] = self.len_observation

        # Ergodic assumption on the empirical distributions
        if not ('ergodic' in self.fit_params.keys()):
            self.fit_params['ergodic'] = False

        # Default Optimizers
        if not ('optimizer_prior' in self.fit_params.keys()):
            self.fit_params['optimizer_prior'] = {'name': 'Adam', 'param': {'lr': 1e-3}}
        if not ('optimizer_inducing_points' in self.fit_params.keys()):
            self.fit_params['optimizer_inducing_points'] = {'name': 'Adam', 'param': {'lr': 1e-3}}
        if not ('optimizer_factors' in self.fit_params.keys()):
            self.fit_params['optimizer_factors'] = {'name': 'Adam', 'param': {'lr': 1e-4}}

        # Logger
        if not ('pct' in self.fit_params.keys()):
            self.fit_params['pct'] = 0.01

        # Fit Prior Mean Function
        if not ('fit_prior_mean' in self.fit_params.keys()):
            self.fit_params['fit_prior_mean'] = True

    def _init_prior_mean_param(self):
        """Initialize the mean parametrization of k=1..K independent prior Gaussian Processes"""

        fit_prior_mean = self.fit_params['fit_prior_mean']
        dim_latent = self.fit_params['dim_latent']
        num_inducing_point = self.num_inducing_point

        # Mean vector ~ dim_latent x num_inducing
        prior_mean_param_tmp = np.zeros((dim_latent, num_inducing_point))
        prior_mean_param = torch.tensor(prior_mean_param_tmp, device=self.device, dtype=self.dtype,
                                        requires_grad=fit_prior_mean)

        # The scale is fixed
        scale_tmp = np.ones(dim_latent)
        scale = torch.tensor(scale_tmp, dtype=self.dtype, device=self.device, requires_grad=False)

        # Lengthscale
        lengthscale_tmp = 0.02 * np.ones(dim_latent)
        lengthscale = torch.tensor(lengthscale_tmp, dtype=self.dtype, device=self.device, requires_grad=fit_prior_mean)

        self.prior_mean_param = (prior_mean_param, scale, lengthscale)

    def _init_kernel(self):
        """ Initialise parameters of k=1..K independent kernels """

        # Number of GP prior
        dim_latent = self.dim_latent

        # Grasp Kernel Type
        kernel_name = self.fit_params['gp_kernel']

        dtype = self.dtype
        device = self.device

        # (Length)scales
        scale = 1 * torch.ones(dim_latent, dtype=dtype, device=device, requires_grad=False)
        lengthscale = 0.01 * torch.ones(dim_latent, dtype=dtype, device=device, requires_grad=False)

        if kernel_name == 'RBF':
            self.prior_covariance_kernel = kernels.RBFKernel(scale, lengthscale)

        elif kernel_name == 'RQ':
            alpha = torch.ones(dim_latent, dtype=dtype, device=device, requires_grad=False)
            self.prior_covariance_kernel = kernels.RQKernel(scale, lengthscale, alpha)

        elif 'Matern' in kernel_name:
            nu = int(kernel_name[-2]) / int(kernel_name[-1])
            self.prior_covariance_kernel = kernels.MaternKernel(scale, lengthscale, nu)

        elif kernel_name == 'Periodic':
            period = 0.1 * torch.ones(dim_latent, dtype=dtype, device=device, requires_grad=False)
            self.prior_covariance_kernel = kernels.PeriodicKernel(scale, lengthscale, period)

        else:
            raise NotImplementedError()

    def _init_recognition(self, observations):
        """ Initialize recognition network of each factor """

        # Outputs diagonal or full covariance matrix
        covariance_type = self.fit_params['constraint_factors']

        # Dimension of the ouptut distributions
        dim_latent = self.dim_latent

        # Tuple of Neural Networks
        recognition_function = ()
        for cur_factor in range(self.num_factors):

            # Neural Network Type
            cur_nn_type = self.fit_params['nn_type'][cur_factor]

            # Number and dimension of each hidden layer
            cur_dim_hidden = self.fit_params['dim_hidden'][cur_factor]

            # Non Linearity Function
            cur_nonlinearity = self.fit_params['nonlinearity'][cur_factor]

            # Input dimension of the current factor
            cur_dim_observation = observations[cur_factor].shape[2:]

            if len(cur_dim_observation) == 1:
                cur_dim_observation = cur_dim_observation[0]

            cur_recognition_function = Net(dim_latent, cur_dim_observation,
                                           nn_type=cur_nn_type,
                                           dim_hidden=cur_dim_hidden,
                                           covariance_type=covariance_type,
                                           nonlinearity=cur_nonlinearity,
                                           ).to(self.device.index)
            recognition_function += (cur_recognition_function,)

        self.recognition_function = recognition_function

    def _init_inducing_points(self):
        """ Initialise the inducing points variational distribution """

        # Setting and dimensions
        dtype = self.dtype
        device = self.device
        dim_latent = self.dim_latent
        num_observation = self.num_observation
        num_inducing_point = self.num_inducing_point
        inducing_locations = self.inducing_locations

        # 1st Natural Parameter
        natural1_tmp = torch.zeros(num_observation, dim_latent, num_inducing_point, dtype=dtype, device=device)
        natural1 = natural1_tmp.clone().detach().requires_grad_(True)

        # 2nd Natural Parameter is initialized using prior
        prior_covariance = self.prior_covariance_kernel(inducing_locations, inducing_locations).detach().clone()
        covariance_tmp = prior_covariance.unsqueeze(0).repeat(self.num_observation, 1, 1, 1)
        Id = diagonalize(torch.ones(num_observation, dim_latent, num_inducing_point, dtype=dtype, device=device))
        natural2_chol = cholesky(0.5 * torch.linalg.inv(covariance_tmp + 1e-3 * Id))
        natural2_chol_vec = tril_to_vector(natural2_chol).clone().detach().requires_grad_(True)

        self.inducing_points_param = (natural1, natural2_chol_vec)
