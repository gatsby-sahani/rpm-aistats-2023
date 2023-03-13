# Imports
import numpy as np

import torch
from torch import matmul
from torch.linalg import cholesky
from torch import log, cholesky_inverse

from torch.distributions.utils import _standard_normal
from torch.distributions.multivariate_normal import _batch_mv as mv


class FlexibleMultivariateNormal:
    """
        Batch multivariate normal distributions parametrised with natural or mean parameters
            Variant of torch.distributions.multivariate_normal.MultivariateNormal designed for RP-GPFA

        The FlexibleMultivariateNormal distributions can be parameterized with:
            - mean vector and (Cholesky) covariance matrix (init_natural=False)
            - 1st and (Cholesky) 2nd natural parameter     (init_natural=True)

        Args:
            param_vector (Tensor): mean or 1st natural parameter
            param_matrix (Tensor): (Cholesky) covariance or 2nd natural parameter

            init_natural (Bool) : Define the parametrization
            init_cholesky (Bool): Full matrix or lower Cholesky

            store_param_chol (Bool): if samples needed, we store scale matrix
            store_suff_stat_mean (Bool): if sufficient statistic mean needed (eg KL(p||.))
            store_suff_stat_variance (Bool): if sufficient statistic variance needed

        Ref:
            https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions
        """

    def __init__(self, param_vector, param_matrix,
                 init_natural=True, init_cholesky=True,
                 store_param_chol=False,
                 store_natural_chol=False,
                 store_suff_stat_mean=False,
                 store_suff_stat_variance=False, jitter=1e-6):

        # Device and Data Type
        self.dtype = param_vector.dtype
        self.device = param_vector.device

        # Check Param Dimensions
        batch_shape_vec = param_vector.shape[:-1]
        batch_shape_mat = param_matrix.shape[:-2]
        event_shape_vec = param_vector.shape[-1]
        event_shape_mat = param_matrix.shape[-2:]

        if not (batch_shape_vec == batch_shape_mat):
            raise ValueError("Incompatible Batch Dimensions")

        if not (event_shape_mat[0] == event_shape_vec) or not (event_shape_mat[1] == event_shape_vec):
            raise ValueError("Incompatible Event Dimensions")

        # Store Valid shapes
        self.batch_shape = batch_shape_vec
        self.event_shape = event_shape_vec

        # Check Param Matrix
        if init_cholesky and not (param_matrix.equal(param_matrix.tril())):
            raise ValueError("Cholesky Matrices must be lower triangular")

        # Jitter Identity Matrix
        Id = self.get_batch_identity(eps=jitter)

        if init_natural:
            # 1st natural parameter
            natural1 = param_vector

            # 2nd natural parameter and its Cholesky Decomposition
            if init_cholesky:
                natural2 = - matmul(param_matrix, param_matrix.transpose(-2, -1))
            else:
                natural2 = param_matrix
                param_matrix = cholesky(-param_matrix + Id)

            # Mean and Covariance Parameters
            covariance = 0.5 * cholesky_inverse(param_matrix)
            mean = matmul(covariance, natural1.unsqueeze(-1)).squeeze(-1)

            # 0.5 * log |covariance|
            t2 = torch.tensor([2], dtype=self.dtype, device=self.device)
            half_log_det_covariance = - torch.log(param_matrix.diagonal(dim1=-2, dim2=-1)).sum(-1) \
                                      - 0.5 * self.event_shape * torch.log(t2)

            if store_param_chol:
                scale = cholesky(covariance)

            if store_natural_chol:
                natural2_chol = param_matrix

        else:
            # With Mean and Covariance parameters
            mean = param_vector

            # Covariance and its Cholesky Decomposition
            if init_cholesky:
                covariance = matmul(param_matrix, param_matrix.transpose(-2, -1))
            else:
                covariance = param_matrix
                param_matrix = cholesky(param_matrix + Id)

            natural2 = - 0.5 * cholesky_inverse(param_matrix)
            natural1 = - 2 * matmul(natural2, mean.unsqueeze(-1)).squeeze(-1)

            # 0.5 * log |covariance|
            half_log_det_covariance = torch.log(param_matrix.diagonal(dim1=-2, dim2=-1)).sum(-1)

            if store_param_chol:
                scale = param_matrix

        # Get and Store Log Normaliser
        log_normaliser = half_log_det_covariance \
                         + 0.5 * matmul(natural1.unsqueeze(-2), mean.unsqueeze(-1)).squeeze(-1).squeeze(-1) \
                         + 0.5 * self.event_shape * log(2 * torch.tensor([np.pi], dtype=self.dtype, device=self.device))
        self.log_normalizer = log_normaliser

        # Get and Store Entropy
        entropy = 0.5 * self.event_shape * (1.0 + log(
            2 * torch.tensor([np.pi], dtype=self.dtype, device=self.device))) + half_log_det_covariance
        self.entropy = entropy

        # Store Natural Parametrization
        self.natural1 = natural1
        self.natural2 = natural2

        if store_param_chol:
            self.scale_tril = scale
            self.loc = mean
        else:
            self.scale_tril = None
            self.loc = None

        if store_natural_chol:
            self.natural2_chol = natural2_chol
        else:
            self.natural2_chol = None

        # Store sufficient Statistics 1st Moment
        if store_suff_stat_mean:
            meanmeanT = matmul(mean.unsqueeze(-1), mean.unsqueeze(-2))
            self.suff_stat_mean = (mean, covariance + meanmeanT)
        else:
            self.suff_stat_mean = None

        # Store sufficient Statistics 2nd Moment
        if store_suff_stat_variance:
            self.suff_stat_variance = get_suff_stat_variance(mean, covariance)
        else:
            self.suff_stat_variance = None

    def mean_covariance(self):
        if self.suff_stat_mean is None:
            raise ValueError("Mean of the sufficient statistic not stored")
        else:
            T1, T2 = self.suff_stat_mean
            mean = T1
            covariance = T2 - matmul(T1.unsqueeze(-1), T1.unsqueeze(-2))
        return mean, covariance

    def suff_stat_vector(self):
        if self.suff_stat_mean is None:
            raise ValueError("Mean of the sufficient statistic not stored")
        else:
            T1, T2 = self.suff_stat_mean

        event_shape = self.event_shape
        batch_shape = self.batch_shape

        return torch.cat((T1, T2.reshape(*batch_shape, event_shape * event_shape)), dim=-1)

    def log_prob(self, value):

        term1 = matmul(self.natural1.unsqueeze(-2), value.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        term2 = matmul(value.unsqueeze(-2), matmul(self.natural2, value.unsqueeze(-1))).squeeze(-1).squeeze(-1)

        return term1 + term2 - self.log_normalizer

    def rsample(self, sample_shape=torch.Size()):

        if (self.scale_tril is None) or (self.loc is None):
            raise ValueError("use_sample set to False")

        shape = sample_shape + self.batch_shape + torch.Size([self.event_shape])
        eps = _standard_normal(shape, dtype=self.dtype, device=self.device)
        return self.loc + mv(self.scale_tril, eps)

    def get_batch_identity(self, eps=1e-6):
        Id = torch.zeros(self.batch_shape + torch.Size([self.event_shape, self.event_shape]),
                         dtype=self.dtype, device=self.device)
        Id[..., :, :] = torch.eye(self.event_shape, dtype=self.dtype, device=self.device)

        eps = torch.tensor(eps, device=self.device, dtype=self.dtype, requires_grad=False)
        # Batch Identity Matrix or 0 tensor
        if eps > 0:
            return eps * Id
        else:
            return eps


class NNFlexibleMultivariateNormal:
    """Build Non-Normalized Flexible Multivariate distributions (Consistent with FlexibleMultivariateNormal)"""
    def __init__(self, natural1, natural2):
        self.natural1 = natural1
        self.natural2 = natural2


def flexible_kl(p: FlexibleMultivariateNormal, q: FlexibleMultivariateNormal,
                repeat1=None, repeat2=None):
    """
    KL divergence between FlexibleMultivariateNormal Distribution using Bregman Divergence Formula

    Args:
        p (FlexibleMultivariateNormal) : 1st distribution (needs sufficient statistics)
        q (FlexibleMultivariateNormal) : 2nd distribution
        repeat1 (list): new dimension id. from p if broadcast needed
        repeat2 (list): new dimension id. from q if broadcast needed
    """

    if p.suff_stat_mean is None:
        raise ValueError("First Distribution needs Sufficient Statistics Moments")

    event_shape_p = p.event_shape
    event_shape_q = q.event_shape

    if not (event_shape_p == event_shape_q):
        raise ValueError("Distribution do not have the same dimensions")

    batch_shape_p = p.batch_shape
    batch_shape_q = q.batch_shape

    if not (batch_shape_p == batch_shape_q):
        # If p and q do not share the same shape: broadcast according to repeat1 and repeat2

        if (repeat1 is None) or (repeat2 is None):
            raise ValueError('Distribution have different batch shape. Must provide helper to combine them')
        else:
            if not (len(batch_shape_p) == len(repeat1)) or not (len(batch_shape_q) == len(repeat2)):
                raise ValueError('Incorrect repeat vector to combine batch shapes.')

            # New Batch Dim
            batch_len = 1 + max([max(repeat1), max(repeat2)])

            # Check that shared dimensions have the same size
            batch_shape1 = torch.zeros(batch_len, dtype=torch.int64, device=p.device)
            batch_shape1[repeat1] = torch.tensor(batch_shape_p, dtype=torch.int64, device=p.device)

            batch_shape2 = torch.zeros(batch_len, dtype=torch.int64, device=p.device)
            batch_shape2[repeat2] = torch.tensor(batch_shape_q, dtype=torch.int64, device=p.device)

            if not (((batch_shape1 - batch_shape2) * batch_shape1 * batch_shape2).count_nonzero() == 0):
                raise ValueError('Incompatible Repeat vectors to combine batch shapes')
            else:

                # New batch Dimensions for p
                batch_shape_p_ext = torch.ones(batch_len, dtype=torch.int64, device=p.device)
                batch_shape_p_ext[repeat1] = torch.tensor(batch_shape_p, dtype=torch.int64, device=p.device)

                # Natural parameters for p
                natural_p_1 = p.natural1.reshape(torch.Size(batch_shape_p_ext)
                                                 + torch.Size([event_shape_p]))
                natural_p_2 = p.natural2.reshape(torch.Size(batch_shape_p_ext)
                                                 + torch.Size([event_shape_p, event_shape_p]))
                natural_p = (natural_p_1, natural_p_2)

                # Log Normalizer for p
                log_normalizer_p = p.log_normalizer.reshape(torch.Size(batch_shape_p_ext))

                # Sufficient Statistic for p
                suff_stat_mean_p_1 = p.suff_stat_mean[0].reshape(torch.Size(batch_shape_p_ext)
                                                                 + torch.Size([event_shape_p]))
                suff_stat_mean_p_2 = p.suff_stat_mean[1].reshape(torch.Size(batch_shape_p_ext)
                                                                 + torch.Size([event_shape_p, event_shape_p]))
                suff_stat_mean_p = (suff_stat_mean_p_1, suff_stat_mean_p_2)

                # New batch Dimensions for q
                batch_shape_q_ext = torch.ones(batch_len, dtype=torch.int64, device=p.device)
                batch_shape_q_ext[repeat2] = torch.tensor(batch_shape_q, dtype=torch.int64, device=p.device)

                # Natural parameters for q
                natural_q_1 = q.natural1.reshape(torch.Size(batch_shape_q_ext)
                                                 + torch.Size([event_shape_q]))
                natural_q_2 = q.natural2.reshape(torch.Size(batch_shape_q_ext)
                                                 + torch.Size([event_shape_q, event_shape_q]))
                natural_q = (natural_q_1, natural_q_2)

                # Log Normalizer for q
                log_normalizer_q = q.log_normalizer.reshape(torch.Size(batch_shape_q_ext))

    else:
        # p and q have the same batch size
        natural_p = (p.natural1, p.natural2)
        natural_q = (q.natural1, q.natural2)

        log_normalizer_p = p.log_normalizer
        log_normalizer_q = q.log_normalizer

        suff_stat_mean_p = p.suff_stat_mean

    return kl(natural_p, natural_q, log_normalizer_p, log_normalizer_q, suff_stat_mean_p)


def kl(natural_p, natural_q, log_normalizer_p, log_normalizer_q, suff_stat_mean_p):
    """
    KL divergence between ExpFam Distribution using Bregman Divergence Formula
    Ref: https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions
    """

    delta_log_normalizer = log_normalizer_q - log_normalizer_p

    delta_natural1 = natural_p[0] - natural_q[0]
    delta_natural2 = natural_p[1] - natural_q[1]

    suff_stat1 = suff_stat_mean_p[0]
    suff_stat2 = suff_stat_mean_p[1]

    term1 = matmul(delta_natural1.unsqueeze(-2), suff_stat1.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    term2 = (delta_natural2 * suff_stat2).sum((-2, -1))

    return delta_log_normalizer + term1 + term2


def vectorize(batch_matrix):
    old_shape = batch_matrix.shape
    new_shape = (*old_shape[:-2], old_shape[-1] * old_shape[-2])
    return batch_matrix.reshape(new_shape)


def kronecker(batch_matrix_a, batch_matrix_b):
    """
    Batch Kronecker product of matrices a and b.
    """

    # Size of the resulting Kroneckered Matrix
    siz1 = torch.Size(torch.tensor(batch_matrix_a.shape[-2:]) * torch.tensor(batch_matrix_b.shape[-2:]))

    # Batch Kronecker Product
    res = batch_matrix_a.unsqueeze(-1).unsqueeze(-3) * batch_matrix_b.unsqueeze(-2).unsqueeze(-4)

    # Batch Size
    siz0 = res.shape[:-4]

    return res.reshape(siz0 + siz1)


def tril_to_vector(batch_matrix):
    """
    Returns N*N Lower Triangular Matrices from an N(N+1) Vector
    """
    n = batch_matrix.shape[-1]
    return batch_matrix[..., np.tril_indices(n)[0], np.tril_indices(n)[1]]


def vector_to_tril(batch_vector):
    """
    Returns N(N+1) Vectors from N*N Lower Triangular Matrices
    """
    # Vector size n(n+1)/2
    n = int(np.floor(-1 / 2 + np.sqrt(1 / 4 + 2 * batch_vector.shape[-1])))

    # Indices
    tri_low_indices = torch.tril_indices(n, n)

    # Init
    batch_matrix = torch.zeros((*batch_vector.shape[:-1], n, n), dtype=batch_vector.dtype, device=batch_vector.device)

    # Set
    batch_matrix[..., tri_low_indices[0], tri_low_indices[1]] = batch_vector

    return batch_matrix


def Aik_Bjl(A, B):
    return kronecker(A, B)


def AilBjk(A, B):
    N = A.shape[-1]
    Gamma = torch.ones(N, 1)
    Atmp = kronecker(Gamma, kronecker(A, Gamma.transpose(-1, -2)))
    Btmp = kronecker(Gamma.transpose(-1, -2), kronecker(B, Gamma))
    return Atmp * Btmp


def Eijkl_EijEkl(sigma, mu):
    mumut = matmul(mu, mu.transpose(-1, -2))
    term1 = Aik_Bjl(sigma, sigma) + AilBjk(sigma, sigma)
    term2 = Aik_Bjl(sigma, mumut) + AilBjk(sigma, mumut)
    term3 = Aik_Bjl(mumut, sigma) + AilBjk(mumut, sigma)
    return term1 + term2 + term3


def Eij_Ek(sigma, mu):
    return kronecker(mu, sigma) + kronecker(sigma, mu)


def get_suff_stat_variance(mu, sigma):

    event_shape = sigma.shape[-1]
    batch_shape = sigma.shape[:-2]
    suff_stat_shape = event_shape * (event_shape + 1)

    C11 = sigma
    C21 = Eij_Ek(sigma, mu.unsqueeze(-1))
    C12 = C21.transpose(-1, -2)
    C22 = Eijkl_EijEkl(sigma, mu.unsqueeze(-1))

    cocovariance = torch.zeros(*batch_shape, suff_stat_shape, suff_stat_shape, dtype=sigma.dtype, device=sigma.device)
    cocovariance[..., :event_shape, :event_shape] = C11
    cocovariance[..., :event_shape, event_shape:] = C12
    cocovariance[..., event_shape:, :event_shape] = C21
    cocovariance[..., event_shape:, event_shape:] = C22

    return cocovariance


def vector_to_tril_diag_idx(n):
    """Returns diagonal indices from the tril vector"""
    return np.cumsum(np.array([0, *np.arange(2, n + 1)]))

