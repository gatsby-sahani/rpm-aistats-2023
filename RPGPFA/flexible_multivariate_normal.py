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
        Batch multivariate normal distributions parametrised with natural parameters
            FlexibleMultivariateNormal is a variant torch.distributions.multivariate_normal.MultivariateNormal designed
            for Recognition Parametrised Gaussian Process Factor Analysis (RP-GPFA)

        The multivariate normal distributions can be parameterized either with:
            - mean vector (and Cholesky decomposition of) covariance matrix (init_natural=False)
            - 1st natural parameter (and Cholesky decomposition of) the 2nd natural parameter (init_natural=True)
        
        Args:
            param_vector (Tensor): mean or 1st natural parameter
            param_matrix (Tensor): (lower Cholesky decomposition of) Covariance matrix or 2nd natural parameter

            init_natural (Bool): Define the parametrization
            init_chol (Bool): Use a lower Cholesky decomposition to parametrize param_matrix

            use_sample (Bool): if rsample needed, we store the lower Cholesky decomposition of covariance matrix
            use_suff_stat_mean (Bool): if sufficient statistic mean needed (eg KL(p||.))
            use_suff_stat_variance (Bool): if sufficient statistic variance needed (eg 2nd order approx)

        Ref: https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions
        """

    def __init__(self, param_vector, param_matrix, init_natural=True, init_chol=True,
                 use_sample=False, use_suff_stat_mean=False, use_suff_stat_variance=False):

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

        self.batch_shape = batch_shape_vec
        self.event_shape = event_shape_vec

        # Check Param Matrix
        if not (param_matrix.equal(param_matrix.tril())) and init_chol:
            raise ValueError("Matrix Parameter must be lower triangular")

        if init_chol:
            if param_matrix.diagonal(dim1=-2, dim2=-1).min() < 0:
                raise ValueError("Cholesky Convention: positive diagonal")

        # Batch Identity Matrix
        Id = self.get_batch_identity()

        # Init Distributions
        if init_natural:
            # With natural Parameters
            natural1 = param_vector

            if init_chol:
                # Cholesky Initialization
                natural2 = - matmul(param_matrix, param_matrix.transpose(-2, -1))
            else:
                # Reparametrize param_matrix with its Cholesky decomposition
                natural2 = param_matrix
                param_matrix = cholesky(-param_matrix)

            covariance = 0.5 * cholesky_inverse(param_matrix + torch.tensor([1e-20], device=self.device) * Id)
            mean = matmul(covariance, natural1.unsqueeze(-1)).squeeze(-1)

            half_log_det_covariance = - (param_matrix.diagonal(dim1=-2, dim2=-1)).log().sum(-1) \
                                      - 0.5 * self.event_shape * torch.log(torch.tensor([2], dtype=self.dtype, device=self.device))
            if use_sample:
                scale_tril = cholesky(covariance)

        else:
            # With parameters
            mean = param_vector

            if init_chol:
                # Cholesky Initialization
                covariance = matmul(param_matrix, param_matrix.transpose(-2, -1))
            else:
                # Reparametrize param_matrix with its Cholesky decomposition
                covariance = param_matrix
                Id = self.get_batch_identity()
                param_matrix = cholesky(param_matrix + 1e-12 * Id)

            natural2 = - 0.5 * cholesky_inverse(param_matrix + torch.tensor([1e-20], device=self.device) * Id)
            natural1 = -2 * matmul(natural2, mean.unsqueeze(-1)).squeeze(-1)

            half_log_det_covariance = param_matrix.diagonal(dim1=-2, dim2=-1).log().sum(-1)

            if use_sample:
                scale_tril = param_matrix

        # Favored Parametrization with naturals
        self.natural1 = natural1
        self.natural2 = natural2

        # Store Normaliser
        log_normaliser = half_log_det_covariance \
                         + 0.5 * matmul(natural1.unsqueeze(-2), mean.unsqueeze(-1)).squeeze(-1).squeeze(-1)  \
                         + 0.5 * self.event_shape * log(2 * torch.tensor([np.pi], dtype=self.dtype, device=self.device))
        self.log_normalizer = log_normaliser

        # Store Entropy
        entropy = 0.5 * self.event_shape * (1.0 + log(2 * torch.tensor([np.pi], dtype=self.dtype, device=self.device))) + half_log_det_covariance
        self.entropy = entropy

        # Store sufficient Statistics 1st Moment
        if use_suff_stat_mean:
            meanmeanT = matmul(mean.unsqueeze(-1), mean.unsqueeze(-2))
            suff_stat_mean = (mean, covariance + meanmeanT)
            self.suff_stat_mean = suff_stat_mean
        else:
            self.suff_stat_mean = None

        # Store sufficient Statistics 2nd Moment
        if use_suff_stat_variance:
            cocovariance = fourth_moment(covariance)
            suff_stat_variance = (covariance, cocovariance)
            self.suff_stat_variance = suff_stat_variance
        else:
            self.suff_stat_variance = None

        if use_sample:
            self.scale_tril = scale_tril
            self.loc = mean
        else:
            self.scale_tril = None
            self.loc = None

    def get_batch_identity(self):
        Id = torch.zeros(self.batch_shape + torch.Size([self.event_shape, self.event_shape]),
                         dtype=self.dtype, device=self.device)
        Id[..., :, :] = torch.eye(self.event_shape, dtype=self.dtype, device=self.device)
        return Id

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

    if not(event_shape_p == event_shape_q):
        raise ValueError("Distribution do not have the same dimensions")

    batch_shape_p = p.batch_shape
    batch_shape_q = q.batch_shape

    if not(batch_shape_p == batch_shape_q):
        # If p and q do not share the same shape: broadcast according to repeat1 and repeat2

        if (repeat1 is None) or (repeat2 is None):
            raise ValueError('Distribution have different batch shape. Must provide helper to combine them')
        else:
            if not(len(batch_shape_p) == len(repeat1)) or not(len(batch_shape_q) == len(repeat2)):
                raise ValueError('Incorrect repeat vector to combine batch shapes.')

            # New Batch Dim
            batch_len = 1+max([max(repeat1), max(repeat2)])

            # Check that shared dimensions have the same size
            batch_shape1 = torch.zeros(batch_len, dtype=torch.int64, device=p.device)
            batch_shape1[repeat1] = torch.tensor(batch_shape_p, dtype=torch.int64, device=p.device)

            batch_shape2 = torch.zeros(batch_len, dtype=torch.int64, device=p.device)
            batch_shape2[repeat2] = torch.tensor(batch_shape_q, dtype=torch.int64, device=p.device)

            if not(((batch_shape1 - batch_shape2) * batch_shape1 * batch_shape2).count_nonzero() == 0):
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


def fourth_moment(batch_covariances):
    """
    Estimate the fourth centered moment of Multivariate Normal with covariance_matrix
    """

    covariance_vector = vectorize(batch_covariances)

    batch_size = batch_covariances.shape[:-2]
    IN1 = torch.ones(batch_size + torch.Size([batch_covariances.shape[-1], 1]),
                     dtype=batch_covariances.dtype, device=batch_covariances.device)
    I1N = IN1.transpose(-2, -1)
    ISI = kronecker(kronecker(IN1, batch_covariances), I1N)

    return xxT(covariance_vector) + kronecker(batch_covariances, batch_covariances) + ISI * ISI.transpose(-1, -2)


def vectorize(batch_matrix):
    old_shape = batch_matrix.shape
    new_shape = (*old_shape[:-2], old_shape[-1]*old_shape[-2])
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


def xxT(batch_vector):
    return matmul(batch_vector.unsqueeze(-1), batch_vector.unsqueeze(-2))


def trace_XXT(batch_matrix):
    n = batch_matrix.size(-1)
    m = batch_matrix.size(-2)
    flat_trace = batch_matrix.reshape(-1, m * n).pow(2).sum(-1)
    return flat_trace.reshape(batch_matrix.shape[:-2])


def vector_to_triul(batch_vector):
    """
    Returns N*N Lower Triangular Matrices from an N(N+1) Vectors
    """

    # Vector size n(n+1)/2
    n = int(np.floor(-1 / 2 + np.sqrt(1 / 4 + 2 * batch_vector.shape[-1])))

    # Get Lower triangular indices of n x n matrix
    tri_low_indices = np.tril_indices(n)

    # tri_low_indices = np.triu_indices(n)
    # L0 = torch.zeros([*batch_vector.shape[:-1], n, n], dtype=batch_vector.dtype, device=batch_vector.device)
    # L0[..., tri_low_indices[0], tri_low_indices[1]] = batch_vector

    # Diagonal Matrix Indices
    diag_indices = (np.arange(n), np.arange(n))

    # Strictly Lower triangular Matrix indices
    tri_low_strict_indices = np.tril_indices(n, k=-1)

    # Indices for on the Cholesky vector (order = F convention !)
    idx_tmp = np.arange(n)
    diag_on_chol = (n * idx_tmp - idx_tmp * (idx_tmp - 1) / 2).astype(int)
    tril_on_chol = np.setxor1d(diag_on_chol, np.arange(batch_vector.shape[-1]))

    # Build lower triangular matrices from data in vectors
    L = torch.zeros([*batch_vector.shape[:-1], n, n], dtype=batch_vector.dtype, device=batch_vector.device)

    # Off diagonal terms
    L[..., tri_low_strict_indices[0], tri_low_strict_indices[1]] = batch_vector[..., tril_on_chol]

    # Diagonal terms constraint to be positive
    L[..., diag_indices[0], diag_indices[1]] = batch_vector[..., diag_on_chol]

    return L




