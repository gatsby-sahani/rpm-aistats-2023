import torch
import numpy as np
from torch.distributions.categorical import Categorical


def rearrange_mnist(train_images, train_labels, num_factors,
                    train_length=60000, sub_ids=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])):
    # Rearange MNIST dataset by grouping num_factors images of with identical labels together

    # Keep Only some digits
    num_digits = len(sub_ids)
    sub_samples_1 = torch.isin(train_labels, sub_ids)
    train_images = train_images[sub_samples_1]
    train_labels = train_labels[sub_samples_1]

    # Sub-Sample and shuffle original dataset
    perm = torch.randperm(len(train_images))
    train_images = train_images[perm[:train_length]]
    train_labels = train_labels[perm[:train_length]]

    # Dimension of each image
    image_size = train_images.shape[-1]

    # Minimum digit occurrence
    num_reps = torch.min(torch.sum(sub_ids.unsqueeze(dim=0) == train_labels.unsqueeze(dim=1), dim=0))
    num_reps = int((np.floor(num_reps / num_factors) * num_factors).squeeze().numpy())

    # Rearranged Datasets: num_reps x num_digits x image_size x image_size
    train_images_factors = torch.zeros((num_reps, num_digits, image_size, image_size))
    train_labels_factors = torch.zeros(num_reps, num_digits)
    for ii in range(len(sub_ids)):
        kept_images = (train_labels == sub_ids[ii])
        train_images_factors[:, ii, :, :] = train_images[kept_images.nonzero()[:num_reps]].squeeze()
        train_labels_factors[:, ii] = train_labels[kept_images.nonzero()[:num_reps]].squeeze()

    # Number of observation per digits
    num_obs_tmp = int(num_reps / num_factors)

    # Rearrange Datasets: num_obs_tmp x num_factors x num_digits x image_size x image_size
    train_images_factors.resize_(num_obs_tmp, num_factors, num_digits, image_size, image_size)
    train_labels_factors.resize_(num_obs_tmp, num_factors, num_digits)

    # Rearrange Datasets: num_obs x num_factors x image_size x image_size
    num_obs = num_obs_tmp * num_digits
    train_images_factors = torch.permute(train_images_factors, (0, 2, 1, 3, 4))
    train_labels_factors = torch.permute(train_labels_factors, (0, 2, 1))

    # train_images_factors.resize_(num_obs, num_factors, image_size, image_size)

    train_images_factors = reshape_fortran(train_images_factors, (num_obs, num_factors, image_size, image_size))
    train_labels_factors = reshape_fortran(train_labels_factors, (num_obs, num_factors))
    train_labels_factors = train_labels_factors[:, 0]

    # Use another Permutation to mix digits
    perm2 = torch.randperm(num_obs)
    train_images_factors = train_images_factors[perm2]
    train_labels_factors = train_labels_factors[perm2]

    observations = [train_images_factors[:, ii] for ii in range(num_factors)]

    # Reshape Training Labels
    train_images_new = train_images_factors.reshape(num_obs * num_factors, image_size, image_size)
    train_labels_new = (train_labels_factors.unsqueeze(dim=1).repeat(1, num_factors)).reshape(
        num_obs * num_factors)

    return observations, train_images_new, train_labels_new


def linear_mapping(x, params):
    # Parametrize a distribution using a linear mapping from x
    # N(params[0]x, diag(params[1]**2))

    N = x.shape[0]
    K = params[0].shape[0]

    variational_mean = torch.matmul(x, torch.transpose(params[0], 0, 1))

    variational_variance = params_to_variance(params[1], data_type=x.dtype)
    variational_variance = (variational_variance.unsqueeze(0)).expand(N, K, K)

    p = torch.distributions.multivariate_normal.MultivariateNormal(variational_mean,
                                                                   covariance_matrix=variational_variance)
    return p


def inv_perm(perm):
    # Inverse a permutation
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


def RBF(x, y, var, length):
    dist = (x.unsqueeze(0)-y.unsqueeze(1))**2/length
    return var * torch.exp(-dist)


def factor_analysis_samples(dim_obs, dim_latent, num_sample, map, covariance_vector, data_type=torch.float32):
    # Generate factor analysis samples :
    # x|z = Cz + N(0, diag(L))
    # dim_obs    : dimension of x
    # dim_latent : dimension of z
    # num_sample : sample number

    Z_mean_prior = np.zeros(dim_latent)
    Z_variance_prior = np.eye(dim_latent)

    latent_sample = np.random.multivariate_normal(Z_mean_prior, Z_variance_prior, size=num_sample)
    obs_sample = latent_sample @ map.transpose() \
              + np.random.multivariate_normal(np.zeros(dim_obs), np.diag(covariance_vector), size=num_sample)

    return torch.tensor(obs_sample, dtype=data_type), torch.tensor(latent_sample, dtype=data_type)


def variance_to_params(variance):
    # Extract A vector containing Cholesky dec. of variance
    dim_oi = variance.shape[0]
    L = torch.linalg.cholesky(variance)
    tri_low_indices = np.tril_indices(dim_oi)

    return L[tri_low_indices]


def params_to_variance(param, data_type=torch.float32):
    # Build Variance Using a 1D cholesky decomposition

    # Vector size n(n+1)/2
    n = int(np.floor(-1 / 2 + np.sqrt(1 / 4 + 2 * param.shape[0])))

    # Get Lower triangular indices of n x n matrix
    tri_low_indices = np.tril_indices(n)

    # Init and fill the cholesky factor
    L = torch.eye(n, dtype=data_type)
    L[tri_low_indices] = param

    return torch.matmul(L, L.transpose(dim0=0, dim1=1))


def stack_observations(observations):
    observations_stacked = torch.stack(observations, dim=1)
    observations_stacked = torch.reshape(observations_stacked,
                                         (observations_stacked.shape[0],
                                          observations_stacked.shape[1] * observations_stacked.shape[2]))
    return observations_stacked


def reshape_fortran(x, shape):
    # Fortran/ Matlab like tensor  reshaping
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def categorical_cross_entropy(p1: Categorical, p2: Categorical):
    # Cross Entropy Between 2 categorical distributions
    min_real = torch.finfo(p2.logits.dtype).min
    logits_p2 = torch.clamp(p2.logits, min=min_real)
    return (p1.probs * logits_p2).sum(-1)


def categorical_kl(p1: Categorical, p2: Categorical):
    # KL Divergences Between 2 categorical distributions
    min_real = torch.finfo(p2.logits.dtype).min
    logits_p1 = torch.clamp(p1.logits, min=min_real)
    logits_p2 = torch.clamp(p2.logits, min=min_real)
    return (p1.probs * logits_p1 - p1.probs * logits_p2).sum(dim=-1)


def categorical_cross_entropy_2D(probs1,probs2):
    min_real = torch.finfo(probs2.dtype).min
    logits2 = torch.clamp(torch.log(probs2), min=min_real)
    return (probs1 * logits2).sum((-1, -2))


def factor_closure(factor_cur):
    def closure(Z):
        fZX = torch.exp(factor_cur.factor.log_prob(Z))
        fZ = torch.mean(fZX, dim=-1, keepdim=True)
        return fZX / (1e-22 + fZ)
    return closure


def factors_closure(factors_tot):
    def closure(Z):
        prod = 1
        for factor_cur in factors_tot:
            fZX = torch.exp(factor_cur.factor.log_prob(Z))
            fZ = torch.mean(fZX, dim=-1, keepdim=True)
            prod *= fZX / (1e-22+ fZ)
        return prod
    return closure


def factors_posterior(factor0, factors_tot, norm=0):
    def closure(Z):
        f = torch.exp(factor0.log_prob(Z))
        for factor_cur in factors_tot:
            fZX = torch.exp(factor_cur.factor.log_prob(Z))
            fZ = torch.mean(fZX, dim=-1, keepdim=True)
            f = f * (fZX / (1e-22 + fZ))

        if norm:
            f = f / (1e-22 + torch.sum(f, dim=0, keepdim=True))
        return f
    return closure
