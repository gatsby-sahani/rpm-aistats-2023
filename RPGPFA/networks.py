# Imports

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
        Neural Network class used to parametrized the recognition potential for RP-GPFA:
        Recognition Parametrised Gaussian Process Factor Analysis

        Outputs the natural parameters of an MVN distributions.

        Args:
            dim_distribution: Dimension of the output distributions
            dim_input: Dimension of the input observation
            dim_hidden: Dimension of each hidden fully connected layer
            nonlinearity: Non linearity function
            full_covariance: Outputs a distribution parameters with full / diagonal covariance
            nn_type: if 'convolutional', 2 conv. layers are added after the input layer
    """

    def __init__(self, dim_distribution, dim_input, dim_hidden=[50, 50],
                 nonlinearity=F.relu, full_covariance=True, nn_type='feedforward'):

        # Output dimensions
        self.full_covariance = full_covariance
        self.dim_distribution = dim_distribution
        self.output_full = \
            int(dim_distribution + dim_distribution * (dim_distribution + 1) / 2)

        if self.full_covariance:
            # Net outputs mean and a full covariance Cholesky Decomposition vector
            dim_output = self.output_full
        else:
            # Net outputs mean and a diagonal Cholesky Decomposition vector
            dim_output = 2 * dim_distribution

        super(Net, self).__init__()

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
            if i == 0:
                self.layers.append(nn.Linear(dim_input, dim_hidden[i]))
            elif i == len(dim_hidden):
                self.layers.append(nn.Linear(dim_hidden[i - 1], dim_output))
            else:
                self.layers.append(nn.Linear(dim_hidden[i - 1], dim_hidden[i]))

        # Id of from Cholesky vector to the diagonal (note that it uses Fortran order convention)
        dim_distribution = self.dim_distribution
        idx_tmp = np.arange(dim_distribution)
        idx_mean = idx_tmp
        idx_diag = (dim_distribution * (1 + idx_tmp) - idx_tmp * (idx_tmp - 1) / 2).astype(int)

        self.nn_type = nn_type
        self.idx_mean = idx_mean
        self.idx_diag = idx_diag
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

        elif self.nn_type == 'feedforward':

            # Feedforward network
            for layer in self.layers[:-1]:
                x = self.nonlinearity(layer(x))
            x = self.layers[-1](x)

        else:

            raise NotImplementedError()

        if self.full_covariance:
            # Diagonal Element must be positive
            x[..., self.idx_diag] = x[..., self.idx_diag] ** 2
            return x
        else:

            y = torch.zeros(x.shape[:-1] + torch.Size([self.output_full]), dtype=x.dtype, device=x.device)
            y[..., self.idx_mean] = x[..., :self.dim_distribution]

            # Diagonal Element must be positive
            y[..., self.idx_diag] = x[..., self.dim_distribution:]**2

            return y

        return x