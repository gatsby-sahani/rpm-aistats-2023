#%% Imports
import numpy as np

import torch
from torch.linalg import inv
from torch import matmul

from scipy.linalg import orthogonal_procrustes
from matplotlib.patches import Ellipse
from utils import diagonalize
import matplotlib.pyplot as plt


def normalize_and_moment_match(logL, Z):

    # Steps
    dZ1 = Z[:, :, 0].diff(dim=0).mean()
    dZ2 = Z[:, :, 1].diff(dim=1).mean()

    # Normalize
    L = torch.exp(logL)
    L /= (L.sum(dim=(-1, -2), keepdim=True) * dZ1 * dZ2)

    # Mean
    mean = ((L.unsqueeze(-1) * Z).sum(dim=(-3, -2)) * dZ1 * dZ2)

    # Variance
    ZZt = matmul(Z.unsqueeze(-1), Z.unsqueeze(-2))
    ZZ0 = matmul(mean.unsqueeze(-1), mean.unsqueeze(-2))
    variance = ((L.unsqueeze(-1).unsqueeze(-1) * ZZt).sum(dim=(-4, -3)) * dZ1 * dZ2)
    variance = variance - ZZ0

    return mean, variance


def posterior_moments(prior_mean, prior_variance, likelihood_mean, likelihood_variance):

    # Jitter
    Id = torch.zeros(likelihood_variance.shape, dtype=likelihood_variance.dtype, device=likelihood_variance.device)
    Id[..., range(Id.shape[-1]), range(Id.shape[-1])] = 1e-6

    # Invert Variance
    prior_variance_inv = inv(prior_variance + Id)
    likelihood_variance_inv = inv(likelihood_variance + Id)

    # Rescale means
    prior_smean = matmul(prior_variance_inv, prior_mean.unsqueeze(-1))
    likelihood_smean = matmul(likelihood_variance_inv, likelihood_mean.unsqueeze(-1))

    # Posterior Moments
    posterior_variance = inv(prior_variance_inv + likelihood_variance_inv)
    posterior_mean = matmul(posterior_variance,  prior_smean + likelihood_smean).squeeze(-1)

    return posterior_mean, posterior_variance


def combine_mvn(mean1, mean2, vari1, vari2, mult=True):

    # Multiply or divide Multivariate Normal Distributions using moments
    if mult:
        sign2 = 1
    else:
        sign2 = -1

    Id = torch.zeros(vari1.shape, dtype=vari1.dtype, device=vari1.device)
    Id[..., range(vari1.shape[-2]), range(vari1.shape[-1])] = 1
    eps = 1e-4

    vari1_inv = inv(vari1 + eps * Id)
    vari2_inv = inv(vari2 + eps * Id)

    vari_new = inv(vari1_inv + sign2 * vari2_inv + eps * Id)
    mean_new = matmul(vari_new,
                      matmul(vari1_inv, mean1.unsqueeze(-1)) + sign2 * matmul(vari2_inv, mean2.unsqueeze(-1))
                      ).squeeze(-1)

    return mean_new, vari_new


def rotate_covariance(covariances):

    # EigenVectors
    L_complex, V_complex = torch.linalg.eig(covariances)

    # Real Part
    L = torch.real(L_complex)
    V = torch.real(V_complex)

    # Max EigenValue
    ev_id = torch.abs(L).argmax(dim=-1)

    # Associated EigenVector
    V = torch.cat([V[nn, :, ev_id[nn]].unsqueeze(0) for nn in range(len(ev_id))])

    # Principal Axis
    alpha = torch.atan(V[:, 1] / V[:, 0]) * 360 / torch.pi

    # Store New Diagonal Of the covariance
    Ld = torch.zeros((L.shape[0], L.shape[1], L.shape[1]), device = L.device, dtype = L.dtype)
    Ld[..., range(L.shape[1]), range(L.shape[1])] = L

    return Ld, alpha


def ellipses_from_stats(mean, covariance):

    # Check that dimensions are compatible
    N = mean.shape[0]
    assert N == covariance.shape[0]

    # Get principal axis of the covariance ellipse
    covariance, rotation_angle = rotate_covariance(covariance)

    # Move to numpy
    mean = mean.detach().numpy()
    covariance = covariance.detach().numpy()

    # Define Rotated and Shifted Ellipses
    ellipses = [Ellipse(
        xy=mean[n],
        width= np.sqrt(covariance[n, 0, 0]) * 2 ,
        height= np.sqrt(covariance[n, 1, 1])* 2 ,
        angle=rotation_angle[n]) for n in range(N)]

    return ellipses


def custom_procurustes(data1, data2):

    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm01 = np.sqrt((mtx1**2).sum(axis=0, keepdims=True))
    norm02 = np.sqrt((mtx2**2).sum(axis=0, keepdims=True))
    mtx1 /= (norm01 + 1e-10)
    mtx2 /= (norm02 + 1e-10)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)
    mtx1 /= (norm1 + 1e-10)
    mtx2 /= (norm2 + 1e-10)

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    R1tot = np.diag((s / (norm01 * norm1)).squeeze())
    R2tot = np.diag((s / (norm02 * norm2)).squeeze()) @ R.T

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return mtx1, mtx2, disparity, R1tot, R2tot


def plot_ellipses(mean, covariance, ax):
    ells = ellipses_from_stats(mean, covariance)

    for e in ells:
        e.zorder = 0
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(1)
        e.set_facecolor([0.5,0.5,0.5])


def linear_regression_1D_latent(latent_true, latent_mean_fit, latent_variance_fit, inducing_mean=None):

    # Dimension of the problem
    num_observation, len_observation, dim_latent = latent_true.shape
    dim_latent_fit = latent_mean_fit.shape[-1]
    shape_true_cur = (num_observation, len_observation, dim_latent)
    shape_true_tmp = (num_observation * len_observation, dim_latent)

    shape_fit_cur = (num_observation, len_observation, dim_latent_fit)
    shape_fit_tmp = (num_observation * len_observation, dim_latent_fit)

    # This linear regression removes degeneraciers only in the 1D case
    assert dim_latent == 1
    #assert dim_latent_fit == 1

    # Reshape And Diagonalize
    latent_true = latent_true[:num_observation].reshape(shape_true_tmp)
    latent_mean_fit = latent_mean_fit.reshape(shape_fit_tmp)
    latent_variance_fit = latent_variance_fit.reshape(shape_fit_tmp)

    # Recenter Rescale True Latent
    latent_true -= latent_true.mean()
    latent_true /= latent_true.max()

    # Recenter Fit
    mean0 = latent_mean_fit.mean(dim=0, keepdim=True)
    latent_mean_fit -= mean0

    # Renormalise Latent
    norm0 = latent_mean_fit.abs().max(dim=0)[0].unsqueeze(0)
    latent_mean_fit /= norm0
    latent_variance_fit /= norm0 ** 2

    # Linear Regression
    Id = diagonalize(torch.ones(dim_latent_fit))
    beta_lr = matmul(inv(matmul(latent_mean_fit.transpose(dim1=-1, dim0=-2), latent_mean_fit) + 0.01 * Id),
                   matmul(latent_mean_fit.transpose(dim1=-1, dim0=-2), latent_true))

    latent_mean_fit = matmul(latent_mean_fit, beta_lr)
    latent_variance_fit = matmul(beta_lr.transpose(dim1=-1, dim0=-2), matmul(diagonalize(latent_variance_fit), beta_lr)).squeeze(-1)

    # norm1 = 1 / matmul(inv(matmul(latent_mean_fit.transpose(dim1=-1, dim0=-2), latent_mean_fit)),
    #                matmul(latent_true.transpose(dim1=-1, dim0=-2), latent_mean_fit))

    # # Renormalise Latent
    # latent_mean_fit /= norm1
    # latent_variance_fit /= norm1 ** 2

    # R square value
    R2 = 1 - ((latent_mean_fit - latent_true)**2).sum() / ((latent_true)**2).sum()

    # Reshape all
    latent_true = latent_true.reshape(shape_true_cur)
    latent_mean_fit = latent_mean_fit.reshape(shape_true_cur)
    latent_variance_fit = latent_variance_fit.reshape(shape_true_cur)

    return latent_true, latent_mean_fit, latent_variance_fit, R2


def plot_gradient_line(xx, cc, **kwargs):
    for tt in range(xx.shape[0]-1):
        plt.plot(xx[tt:tt+2, 0], xx[tt:tt+2,  1], c=cc[tt], **kwargs)
