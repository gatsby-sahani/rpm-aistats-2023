# Imports

import torch
import numpy as np
import torch.nn as nn
from torch import matmul
from utils import diagonalize

__all__ = ['Kernel', 'RBFKernel', 'RQKernel', 'MaternKernel', 'PeriodicKernel']


def squared_euclidian_distance(locations1, locations2):
    """Distances between locations"""
    # locations1 ~ N1 x D locations2 ~ N2 x D

    # locations1 - locations2 ~ N1 x N2 x D
    diff = locations1.unsqueeze(-2) - locations2.unsqueeze(-3)

    return matmul(diff.unsqueeze(-2), diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)


class Kernel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, locations1, locations2):
        raise NotImplementedError

    def posteriors(self, locations1, locations2):
        """ Kernel Posterior helpers """

        # Cov_k(M, M) ~ K x M x M
        K_MM = self.forward(locations1, locations1)

        # Identity matrix ~ K x M x M
        Id = 1e-6 * diagonalize(torch.ones(K_MM.shape[:-1], device=locations1.device, dtype=locations1.dtype))

        # inv Cov_k(M, M) ~ K x M x M
        K_MM_inv = torch.linalg.inv(K_MM + Id)

        # Cov_k(T, M) ~ K x T x M
        K_TM = self.forward(locations2, locations1)

        # Cov_k(t,t) ~ K x T (we only keep the diagonal elements)
        K_T = self.forward(locations2, locations2).diagonal(dim1=-1, dim2=-2)

        # Cov_k(T, M) inv( Cov_k(M, M) ) unsqueezed to ~ K x T x M
        K_TM_K_MM_inv = matmul(K_TM, K_MM_inv)

        return K_T, K_MM, K_MM_inv, K_TM, K_TM_K_MM_inv


class RBFKernel(Kernel):
    """Exponentiated quadratic kernel"""
    def __init__(self, scale, lengthscale, copy_scale=False, copy_lenghscale=True,  **kwargs):
        super().__init__(**kwargs)

        self.scale = nn.Parameter(scale, requires_grad=copy_scale)
        self.lengthscale = nn.Parameter(lengthscale, requires_grad=True) if copy_lenghscale else lengthscale

    def forward(self, locations1, locations2):
        # ||locations1 - locations2||^2 ~ 1 x N1 x N2
        sdist = squared_euclidian_distance(locations1, locations2).unsqueeze(0)

        # Expand and square
        scale_expanded = (self.scale ** 2).unsqueeze(-1).unsqueeze(-1)
        lengthscale_expanded = (self.lengthscale ** 2).unsqueeze(-1).unsqueeze(-1)

        # K(locations1, locations2)
        K = scale_expanded * torch.exp(- 0.5 * sdist / lengthscale_expanded)

        return K


class RQKernel(Kernel):
    """Rational quadratic kernel"""
    def __init__(self, scale, lengthscale, alpha, copy_scale=False, copy_lenghscale=True, copy_alpha=True, **kwargs):
        super().__init__(**kwargs)

        self.scale = nn.Parameter(scale, requires_grad=True) if copy_scale else scale
        self.alpha = nn.Parameter(alpha, requires_grad=True) if copy_alpha else alpha
        self.lengthscale = nn.Parameter(lengthscale, requires_grad=True) if copy_lenghscale else lengthscale

    def forward(self, locations1, locations2):
        # ||locations1 - locations2||^2 ~ 1 x N1 x N2
        sdist = squared_euclidian_distance(locations1, locations2).unsqueeze(0)

        # Expand and square
        scale_expanded = (self.scale ** 2).unsqueeze(-1).unsqueeze(-1)
        alpha_expanded = (self.alpha ** 2).unsqueeze(-1).unsqueeze(-1)
        lengthscale_expanded = (self.lengthscale ** 2).unsqueeze(-1).unsqueeze(-1)

        # K(locations1, locations2)
        K = scale_expanded * (1 + sdist / (2 * alpha_expanded * lengthscale_expanded)) ** (-alpha_expanded)

        return K


class MaternKernel(Kernel):
    """Matern kernel"""
    def __init__(self, scale, lengthscale, nu, copy_scale=False, copy_lenghscale=True, **kwargs):
        super().__init__(**kwargs)
        self.nu = nu
        self.scale = nn.Parameter(scale, requires_grad=True) if copy_scale else scale
        self.lengthscale = nn.Parameter(lengthscale, requires_grad=True) if copy_lenghscale else lengthscale

    def forward(self, locations1, locations2):
        # ||locations1 - locations2||^2 ~ 1 x N1 x N2
        sdist = squared_euclidian_distance(locations1, locations2).unsqueeze(0)

        # Expand and square
        nu = self.nu
        scale_expanded = (self.scale ** 2).unsqueeze(-1).unsqueeze(-1)
        lengthscale_expanded = (self.lengthscale ** 2).unsqueeze(-1).unsqueeze(-1)

        # Exponential Term
        expd = torch.exp(-torch.sqrt(sdist * nu) / lengthscale_expanded)

        # Order dependent term
        if nu == 1:
            cons = 1
        elif nu == 3:
            cons = 1 + torch.sqrt(sdist * nu) / lengthscale_expanded
        elif nu == 5:
            cons = 1 + torch.sqrt(sdist * nu) / lengthscale_expanded + sdist * nu / (3 * lengthscale_expanded ** 2)

        # K(locations1, locations2)
        K = scale_expanded * cons * expd

        return K


class PeriodicKernel(Kernel):
    """Periodic Kernel"""
    def __init__(self, scale, lengthscale, period, copy_scale=False, copy_lenghscale=True, copy_alpha=True, **kwargs):
        super().__init__(**kwargs)
        self.scale = nn.Parameter(scale, requires_grad=True) if copy_scale else scale
        self.period = nn.Parameter(period, requires_grad=True) if copy_alpha else period
        self.lengthscale = nn.Parameter(lengthscale, requires_grad=True) if copy_lenghscale else lengthscale

    def forward(self, locations1, locations2):
        # ||locations1 - locations2||^2 ~ 1 x N1 x N2
        dist = torch.sqrt(squared_euclidian_distance(locations1, locations2)).unsqueeze(0)

        # Expand and square
        period_expanded = (self.period ** 2).unsqueeze(-1).unsqueeze(-1)
        scale_expanded = (self.scale  ** 2).unsqueeze(-1).unsqueeze(-1)
        lengthscale_expanded = (self.lengthscale ** 2).unsqueeze(-1).unsqueeze(-1)

        K = scale_expanded * torch.exp(- 2 * torch.sin(dist * np.pi / period_expanded) ** 2 / lengthscale_expanded)

        return K


class POLYKernel(Kernel):
    """Poly kernel"""
    def __init__(self, alpha0=1, dim0=3, **kwargs):
        super().__init__(**kwargs)
        self.alpha0 = nn.Parameter(alpha0, requires_grad=True)
        self.dim0 = nn.Parameter(dim0, requires_grad=True)

    def forward(self, locations1, locations2):
        # ||locations1 - locations2||^2 ~ 1 x N1 x N2
        sprod = matmul(locations1, locations2.transpose(-1, -2)).unsqueeze(0)

        # Expand and square
        alpha_expanded = (self.alpha0 ** 2).unsqueeze(-1).unsqueeze(-1)
        dim0_expanded = (self.dim0 ** 2).unsqueeze(-1).unsqueeze(-1)

        # K(locations1, locations2)
        K = (sprod + alpha_expanded) ** dim0_expanded

        return K









