"""
Script is used to check our custom flexible_multivariate_normal and its agreement with torch.multivariate_normal
"""

# Imports
import torch
import numpy as np
from torch.linalg import matmul
from matplotlib import pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from flexible_multivariate_normal import FlexibleMultivariateNormal, flexible_kl, \
    vector_to_tril, tril_to_vector,  Eij_Ek, Eijkl_EijEkl, get_suff_stat_variance


# Seeds
np.random.seed(10)
torch.manual_seed(10)

# Manually set jitter
eps = 1e-12

# Dimensions
event_shape = 4
batch_shape = [2, 3]
ddtype = torch.float64
chol_param_number = int(event_shape * (event_shape + 1) / 2)

# First Set of Parameters
chol_vector_a = 0.25 + torch.rand(*batch_shape, chol_param_number, dtype=ddtype)
chol_matrix_a = vector_to_tril(chol_vector_a)
means_a = torch.rand((*chol_vector_a.shape[:-1], event_shape), dtype=ddtype)

# Second Set of Parameters
chol_vector_b = 0.25 + torch.rand(*batch_shape, chol_param_number, dtype=ddtype)
chol_matrix_b = vector_to_tril(chol_vector_b)
means_b = torch.rand((*chol_vector_b.shape[:-1], event_shape), dtype=ddtype)

# Flexible MultiVariateNormal
mvn_a_flexible = FlexibleMultivariateNormal(means_a, chol_matrix_a,
                                            init_natural=False, store_suff_stat_mean=True,
                                            store_suff_stat_variance=True, store_param_chol=True, jitter=eps)
mvn_b_flexible = FlexibleMultivariateNormal(means_b, chol_matrix_b,
                                            init_natural=False, store_suff_stat_mean=True,
                                            store_suff_stat_variance=True, store_param_chol=True, jitter=eps)

# Standard MultiVariateNormal
mvn_a_standard = MultivariateNormal(means_a, scale_tril=chol_matrix_a)
mvn_b_standard = MultivariateNormal(means_b, scale_tril=chol_matrix_b)

# Check Log Prob
value_shape = [10, 10] + [1 for _ in range(len(batch_shape))] + [event_shape]
value = torch.randn(value_shape, dtype=ddtype)

log_prob_flexible = mvn_a_flexible.log_prob(value)
log_prob_standard = mvn_a_standard.log_prob(value)
dlog_prob = torch.sum(torch.abs((log_prob_flexible - log_prob_standard)))

if dlog_prob < 1e-4:
    print("Passed: log_prob")
else:
    print('Failed test')

# Check KL divergences
kl_standard = torch.distributions.kl_divergence(mvn_a_standard, mvn_b_standard)
kl_flexible = flexible_kl(mvn_a_flexible, mvn_b_flexible)
dkl = torch.sum(torch.abs((kl_flexible - kl_standard)))
if dkl < 1e-4:
    print("Passed: KL")
else:
    print('Failed test')

# Check Entropy
entropy_flexible = mvn_a_flexible.entropy
entropy_standard = mvn_a_standard.entropy()
dentopy = torch.sum(torch.abs((entropy_flexible - entropy_standard)))
if dentopy < 1e-4:
    print("Passed: Entropy")
else:
    print('Failed test')

# Check transition natural to parameters and vice versa
mvn_a_naturals = FlexibleMultivariateNormal(mvn_a_flexible.natural1,
                                            torch.linalg.cholesky(-mvn_a_flexible.natural2),
                                            init_natural=True, store_param_chol=True,
                                            store_suff_stat_mean=True, store_suff_stat_variance=True, jitter=eps)
means_a_from_natural = mvn_a_naturals.loc
chol_matrix_a_from_natural = mvn_a_naturals.scale_tril

dmeans = torch.sum(torch.abs((means_a_from_natural - means_a)))
dchol_matrix = torch.sum(torch.abs((chol_matrix_a_from_natural - chol_matrix_a)))

mean0 = mvn_a_naturals.suff_stat_mean[0]
mmT = torch.linalg.matmul(mean0.unsqueeze(-1), mean0.unsqueeze(-2))
Sigma0 = mvn_a_naturals.suff_stat_mean[1] - mmT
Sigma1 = mvn_a_naturals.suff_stat_variance[..., :event_shape, :event_shape ]

dchol_covariance = torch.sum(torch.abs(torch.linalg.cholesky(Sigma1) - chol_matrix_a))
dcovariance = torch.sum(torch.abs((mvn_a_standard.covariance_matrix - Sigma1)))

if (dmeans < 1e-6) & (dchol_matrix < 1e-6) & (dchol_matrix < 1e-6):
    print("Passed: Natural to Param and Param to Natural")
else:
    print('Failed test')

# Check rsample
sample_shape = torch.Size([100])
sample_a_standard = mvn_a_standard.rsample(sample_shape)
sample_a_flexible = mvn_a_naturals.rsample(sample_shape)

x_test = 0
y_test = 1

sample_a_standard_test = sample_a_standard[:, x_test, y_test].squeeze()
sample_a_flexible_test = sample_a_flexible[:, x_test, y_test].squeeze()

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.scatter(sample_a_standard_test[:, 0], sample_a_standard_test[:, 1])
plt.title('Standard Samples')

plt.subplot(1, 2, 2)
plt.scatter(sample_a_flexible_test[:, 0], sample_a_flexible_test[:, 1])
plt.title('Flexible Samples')

# Init with full matrices instead of Cholesky decomposition
mvn_a_flexible_mat = FlexibleMultivariateNormal(means_a, matmul(chol_matrix_a, chol_matrix_a.transpose(-2, -1)),
                                                init_natural=False, init_cholesky=False, store_suff_stat_mean=True,
                                                store_suff_stat_variance=True, store_param_chol=True, jitter=eps)
mvn_a_naturals_mat = FlexibleMultivariateNormal(mvn_a_flexible.natural1,
                                                mvn_a_flexible.natural2,
                                                init_natural=True, init_cholesky=False, store_param_chol=True,
                                                store_suff_stat_mean=True, store_suff_stat_variance=True, jitter=eps)

d = torch.zeros([11])
d[0] = torch.sum(torch.sum(mvn_a_flexible.natural1 - mvn_a_flexible_mat.natural1))
d[1] = torch.sum(torch.sum(mvn_a_flexible.natural2 - mvn_a_flexible_mat.natural2))
d[2] = torch.sum(torch.sum(mvn_a_flexible.entropy - mvn_a_flexible_mat.entropy))
d[3] = torch.sum(torch.sum(mvn_a_flexible.suff_stat_mean[0] - mvn_a_flexible_mat.suff_stat_mean[0]))
d[4] = torch.sum(torch.sum(mvn_a_flexible.suff_stat_mean[1] - mvn_a_flexible_mat.suff_stat_mean[1]))
d[5] = torch.sum(torch.sum(mvn_a_flexible.suff_stat_variance[0] - mvn_a_flexible_mat.suff_stat_variance[0]))
d[6] = torch.sum(torch.sum(mvn_a_flexible.suff_stat_variance[1] - mvn_a_flexible_mat.suff_stat_variance[1]))
d[7] = torch.sum(torch.sum(mvn_a_flexible.loc - mvn_a_flexible_mat.loc))
d[8] = torch.sum(torch.sum(mvn_a_flexible.scale_tril - mvn_a_flexible_mat.scale_tril))
d[9] = torch.sum(torch.sum(mvn_a_flexible.log_normalizer - mvn_a_flexible_mat.log_normalizer))
d[10] = torch.sum(torch.sum(mvn_a_flexible.log_prob(value) - mvn_a_flexible_mat.log_prob(value)))

l = torch.zeros([11])
l[0] = torch.sum(torch.sum(mvn_a_flexible.natural1 - mvn_a_naturals_mat.natural1))
l[1] = torch.sum(torch.sum(mvn_a_flexible.natural2 - mvn_a_naturals_mat.natural2))
l[2] = torch.sum(torch.sum(mvn_a_flexible.entropy - mvn_a_naturals_mat.entropy))
l[3] = torch.sum(torch.sum(mvn_a_flexible.suff_stat_mean[0] - mvn_a_naturals_mat.suff_stat_mean[0]))
l[4] = torch.sum(torch.sum(mvn_a_flexible.suff_stat_mean[1] - mvn_a_naturals_mat.suff_stat_mean[1]))
l[5] = torch.sum(torch.sum(mvn_a_flexible.suff_stat_variance[0] - mvn_a_naturals_mat.suff_stat_variance[0]))
l[6] = torch.sum(torch.sum(mvn_a_flexible.suff_stat_variance[1] - mvn_a_naturals_mat.suff_stat_variance[1]))
l[7] = torch.sum(torch.sum(mvn_a_flexible.loc - mvn_a_naturals_mat.loc))
l[8] = torch.sum(torch.sum(mvn_a_flexible.scale_tril - mvn_a_naturals_mat.scale_tril))
l[9] = torch.sum(torch.sum(mvn_a_flexible.log_normalizer - mvn_a_naturals_mat.log_normalizer))
l[10] = torch.sum(torch.sum(mvn_a_flexible.log_prob(value) - mvn_a_naturals_mat.log_prob(value)))

if (d.max() < 1e-5) and (l.max() < 1e-5):
    print('Passed: Init with matrix or cholseky(matrix)')
else:
    print('Failed test')

# Broadcast KL divergence when distributions do not have the same mean
# Dimensions
event_shape_c = 2
batch_shape_c = [4, 3, 5]

# First Set of Parameters
chol_vector_c = 0.25 + torch.rand(*batch_shape_c, chol_param_number, dtype=ddtype)
chol_matrix_c = vector_to_tril(chol_vector_c)
means_c = torch.rand((*chol_vector_c.shape[:-1], event_shape), dtype=ddtype)

# Flexible MultiVariateNormal
mvn_c_flexible = FlexibleMultivariateNormal(means_c, chol_matrix_c,
                                            init_natural=False, store_suff_stat_mean=True,
                                            store_suff_stat_variance=True, store_param_chol=True, jitter=eps)

broadcasted_kl = flexible_kl(mvn_a_flexible, mvn_c_flexible, repeat1=[0, 2], repeat2=[1, 2, 3])

non_broadcasted_kl = torch.zeros(broadcasted_kl.shape)
for ii in range(means_a.shape[0]):
    for jj in range(means_c.shape[0]):
        for kk in range(means_a.shape[1]):
            for ll in range(means_c.shape[2]):
                means_a_cur = means_a[ii, kk]
                chol_matrix_a_cur = chol_matrix_a[ii, kk]

                means_c_cur = means_c[jj, kk, ll]
                chol_matrix_c_cur = chol_matrix_c[jj, kk, ll]

                mvn_a_cur = FlexibleMultivariateNormal(means_a_cur, chol_matrix_a_cur,
                                                       init_natural=False, init_cholesky=True,
                                                       store_suff_stat_mean=True, jitter=eps)
                mvn_c_cur = FlexibleMultivariateNormal(means_c_cur, chol_matrix_c_cur,
                                                       init_natural=False, init_cholesky=True,
                                                       store_suff_stat_mean=False, jitter=eps)
                non_broadcasted_kl[ii, jj, kk, ll] = flexible_kl(mvn_a_cur, mvn_c_cur)

if torch.sum(torch.abs(non_broadcasted_kl - broadcasted_kl)) < 1e-4:
    print('Passed: KL broadcasting')
else:
    print('Failed test')

# Compare init with natural/param, cholesky/full and cholesky/eigendecomposition
mean = means_a
covariance_cholesky = chol_matrix_a
covariance = matmul(chol_matrix_a, chol_matrix_a.transpose(-1, -2))

natural2 = - 0.5 * torch.cholesky_inverse(chol_matrix_a)
natural2_cholesky = torch.linalg.cholesky(-natural2)
natural1 = - 2 * matmul(natural2, means_a.unsqueeze(-1)).squeeze(-1)

reference_distribution = FlexibleMultivariateNormal(means_a, chol_matrix_a,
                                                    init_natural=False, init_cholesky=True,
                                                    store_param_chol=True,
                                                    store_suff_stat_mean=True,
                                                    store_suff_stat_variance=True, jitter=eps)

parameters_full_vanilla = {'p1': mean, 'p2': covariance,
                           'init_natural': False, 'init_cholesky': False,
                           'store_eigen_decomposition': False}
parameters_full_eigen = {'p1': mean, 'p2': covariance,
                         'init_natural': False, 'init_cholesky': False,
                         'store_eigen_decomposition': True}
parameters_chol_vanilla = {'p1': mean, 'p2': covariance_cholesky,
                           'init_natural': False, 'init_cholesky': True,
                           'store_eigen_decomposition': False}
parameters_chol_eigen = {'p1': mean, 'p2': covariance_cholesky,
                         'init_natural': False, 'init_cholesky': True,
                         'store_eigen_decomposition': True}
natural_full_vanilla = {'p1': natural1, 'p2': natural2,
                        'init_natural': True, 'init_cholesky': False,
                        'store_eigen_decomposition': False}
natural_full_eigen = {'p1': natural1, 'p2': natural2,
                      'init_natural': True, 'init_cholesky': False,
                      'store_eigen_decomposition': True}
natural_chol_vanilla = {'p1': natural1, 'p2': natural2_cholesky,
                        'init_natural': True, 'init_cholesky': True,
                        'store_eigen_decomposition': False}
natural_chol_eigen = {'p1': natural1, 'p2': natural2_cholesky,
                      'init_natural': True, 'init_cholesky': True,
                      'store_eigen_decomposition': True}

all_parametrisation = [parameters_full_vanilla, parameters_full_eigen, parameters_chol_vanilla, parameters_chol_eigen,
                       natural_full_vanilla, natural_full_eigen, natural_chol_vanilla, natural_chol_eigen]

for id, pc in enumerate(all_parametrisation):
    distribution_cur = FlexibleMultivariateNormal(pc['p1'], pc['p2'],
                                                  init_natural=pc['init_natural'], init_cholesky=pc['init_cholesky'],
                                                  store_param_chol=True,
                                                  store_suff_stat_mean=True,
                                                  store_suff_stat_variance=True, jitter=eps)

    d1 = ((distribution_cur.natural1 - reference_distribution.natural1) ** 2).sum() \
         / ((reference_distribution.natural1) ** 2).sum()

    d2 = ((distribution_cur.natural2 - reference_distribution.natural2) ** 2).sum() \
         / ((reference_distribution.natural2) ** 2).sum()

    if (d1 + d2) < 1e-4:
        print('Passed: Test Param ' + str(id + 1) + '/' + str(len(all_parametrisation)))
    else:
        print('Failed Test Param ' + str(id + 1) + '/' + str(len(all_parametrisation)))


n = 7
batch_vec0 = torch.rand(30, 40, int(n *(n+1) / 2)) - 0.5
batch_mat0 = vector_to_tril(batch_vec0)
batch_vec1 = tril_to_vector(batch_mat0)
batch_mat1 = vector_to_tril(batch_vec1)

if (((batch_vec0 - batch_vec1)**2).sum() + ((batch_mat0 - batch_mat1)**2).sum()) < 1e-12:
    print('Passed: Triangulation')
else:
    print('Failed test')

#% Check co-covariance
batch_shape = 3
event_shape = 5

mu = torch.rand(batch_shape, event_shape)
tmp = torch.rand(batch_shape, event_shape, event_shape)
sigma = torch.matmul(tmp, tmp.transpose(-1,-2))

L = torch.zeros(batch_shape, event_shape * event_shape, event_shape)
for ii in range(event_shape):
    for jj in range(event_shape):
        for kk in range(event_shape):
            mm = ii + jj * event_shape
            L[:, mm, kk] = mu[:, ii] * sigma[:, jj, kk] + mu[:, jj] * sigma[:, ii, kk]


S = torch.zeros(batch_shape, event_shape * event_shape, event_shape * event_shape)
gamma = torch.ones(event_shape, 1)
for ii in range(event_shape):
    for jj in range(event_shape):
        for kk in range(event_shape):
            for ll in range(event_shape):
                nn = ii + jj * event_shape
                mm = kk + ll * event_shape

                S[:, nn, mm] = sigma[:, ii, kk] * sigma[:, jj, ll] + sigma[:, ii, ll] * sigma[:, jj, kk] \
                            + sigma[:, ii, kk] * mu[:, jj] * mu[:, ll] \
                            + sigma[:, ii, ll] * mu[:, jj] * mu[:, kk] \
                            + sigma[:, jj, kk] * mu[:, ii] * mu[:, ll] \
                            + sigma[:, jj, ll] * mu[:, ii] * mu[:, kk]

Lfast = Eij_Ek(sigma, mu.unsqueeze(-1))
Sfast = Eijkl_EijEkl(sigma, mu.unsqueeze(-1))

if (L - Lfast).abs().sum() / L.abs().sum() < 1e-6:
    print('Passed: L test')
else:
    print('Failed test')

if (S - Sfast).abs().sum() / S.abs().sum() < 1e-6:
    print('Passed: S test')
else:
    print('Failed test')


def get_suff_stat(x):
    xxt = matmul(x.unsqueeze(-1), x.unsqueeze(-2)).reshape( (*x.shape[:-1], x.shape[-1]**2))
    return torch.cat((x, xxt), dim=-1)

vari_theory = get_suff_stat_variance(mu, sigma)
dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu.squeeze(-1), covariance_matrix=sigma)
num_samples = 100000
samples = dist.sample(torch.tensor([num_samples]))
tx = get_suff_stat(samples)
mean_sample = tx.sum(dim=0) / num_samples
vari_sample = matmul((tx-mean_sample).unsqueeze(-1), (tx-mean_sample).unsqueeze(-2)).sum(dim=0) / (num_samples -1)
dist1 = (vari_sample - vari_theory).abs().sum() / (vari_sample).abs().sum()

print('Relative Distance ' + str(num_samples) + ' samples: ' + str(dist1))



plt.figure()
for nn in range(batch_shape):

    plt.subplot(batch_shape, 2, 2*nn + 1)
    plt.imshow(vari_theory[nn])
    plt.title('Theory')
    plt.subplot(batch_shape, 2, 2*nn + 2)
    plt.imshow(vari_sample[nn])
    plt.title('Samples')

