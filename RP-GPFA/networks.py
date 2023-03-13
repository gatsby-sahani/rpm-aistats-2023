# Imports

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from flexible_multivariate_normal import vector_to_tril_diag_idx


class Net(nn.Module):
    """
        Neural Network used to parametrized the recognition potential for RP-GPFA:
        It outputs the natural parameters of an MVN distributions.

        Args:
            dim_distribution: Dimension of the output distributions
            dim_input: Dimension of the input observation
            dim_hidden: Dimension of each hidden fully connected layer
            nonlinearity: Non linearity function
            covariance_type: Outputs a distribution parameters with full / diagonal  / fixed / fixed_diag covariance
            nn_type: default 'perceptron'. if 'convolutional', 2 conv. layers are added after the input layer
    """

    def __init__(self, dim_distribution, dim_input, dim_hidden=[50, 50],
                 nonlinearity=F.relu, covariance_type='full', nn_type='perceptron'):

        super(Net, self).__init__()

        # Output dimensions
        self.covariance_type = covariance_type
        self.dim_distribution = dim_distribution
        self.output_full = int(dim_distribution * (dim_distribution + 3) / 2)

        if self.covariance_type == 'full':
            # Net outputs mean and a full covariance Cholesky Decomposition vector
            dim_output = self.output_full

        elif self.covariance_type == 'diag':
            # Net outputs mean and a diagonal Cholesky Decomposition vector
            dim_output = 2 * dim_distribution

        elif self.covariance_type == 'fixed':
            dim_output = dim_distribution
            diag_idx = vector_to_tril_diag_idx(dim_distribution)
            bias_init = torch.zeros(int(dim_distribution * (dim_distribution + 1) / 2), requires_grad=False)
            bias_init[diag_idx] = -0.5
            self.bias = torch.nn.Parameter(bias_init, requires_grad=True)

        elif self.covariance_type == 'fixed_diag':
            dim_output = dim_distribution
            bias_init = -0.5 * torch.ones(dim_distribution, requires_grad=False) # TODO : remove this line !
            self.bias = torch.nn.Parameter(bias_init, requires_grad=True)

        self.layers = nn.ModuleList()

        if nn_type == 'convolutional':

            # Convolutional Layers kernel
            kernel_size = 5

            # Output size after convolution and pooling
            conv_output_x = ((dim_input[0] - kernel_size + 1) / 2 - kernel_size + 1) / 2
            conv_output_y = ((dim_input[1] - kernel_size + 1) / 2 - kernel_size + 1) / 2

            # Convolutional Layers
            self.layers.append(nn.Conv2d(1, 10, kernel_size=kernel_size))
            self.layers.append(nn.Conv2d(10, 20, kernel_size=kernel_size))

            # Linearized + Collapse channels
            dim_input = int(20 * conv_output_x * conv_output_y)

        # Build Feedforward net with dim_hidden
        for i in range(len(dim_hidden) + 1):
            if len(dim_hidden) > 0:
                if i == 0:
                    self.layers.append(nn.Linear(dim_input, dim_hidden[i]))
                elif i == len(dim_hidden):
                    self.layers.append(nn.Linear(dim_hidden[i - 1], dim_output))
                else:
                    self.layers.append(nn.Linear(dim_hidden[i - 1], dim_hidden[i]))
            else:
                self.layers.append(nn.Linear(dim_input, dim_output))

        # Id of from Cholesky vector to the diagonal (note that it uses Fortran order convention)
        dim_distribution = self.dim_distribution

        self.nn_type = nn_type
        self.idx_mean = np.arange(dim_distribution)
        self.idx_diag = dim_distribution + vector_to_tril_diag_idx(dim_distribution)
        self.dim_input = dim_input
        self.nonlinearity = nonlinearity

    def forward(self, x):

        if self.nn_type == 'convolutional':

            # Convolutional Layers + Pooling
            x = self.nonlinearity(F.max_pool2d(self.layers[0](x), 2))
            x = self.nonlinearity(F.max_pool2d(self.layers[1](x), 2))
            x = x.view(-1, self.dim_input)

            # Layers
            for layer in self.layers[2:-1]:
                x = self.nonlinearity(layer(x))
            x = self.layers[-1](x)

        elif self.nn_type == 'perceptron':

            # Feedforward network
            for layer in self.layers[:-1]:
                x = self.nonlinearity(layer(x))
            x = self.layers[-1](x)

        else:
            raise NotImplementedError()

        if self.covariance_type == 'full':
            return x

        elif self.covariance_type == 'diag':
            y = torch.zeros(x.shape[:-1] + torch.Size([self.output_full]), dtype=x.dtype, device=x.device)
            y[..., self.idx_mean] = x[..., :self.dim_distribution]
            y[..., self.idx_diag] = x[..., self.dim_distribution:]
            return y

        elif self.covariance_type == 'fixed':
            y = torch.zeros(x.shape[:-1] + torch.Size([self.output_full]), dtype=x.dtype, device=x.device)
            y[..., :x.shape[-1]] = x
            y[..., x.shape[-1]:] = self.bias
            return y

        elif self.covariance_type == 'fixed_diag':

            y = torch.zeros(x.shape[:-1] + torch.Size([self.output_full]), dtype=x.dtype, device=x.device)
            y[..., :x.shape[-1]] = x
            y[..., self.idx_diag] = self.bias
            return y

