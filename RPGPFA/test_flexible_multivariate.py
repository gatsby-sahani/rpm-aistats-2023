"""
Script is used to check the agreement between torch.multivariate_normal and our custom flexible_multivariate_normal
"""

# Imports
import torch
import numpy as np
from torch.linalg import matmul
from torch.distributions.multivariate_normal import MultivariateNormal
from flexible_multivariate_normal import FlexibleMultivariateNormal, flexible_kl, vector_to_triul


from matplotlib import pyplot as plt

# Seeds
np.random.seed(1)
torch.manual_seed(1)

# Dimensions
event_shape = 2
batch_shape = [2, 3]
ddtype = torch.float64
chol_param_number = int(event_shape * (event_shape +1) / 2)

# First Set of Parameters
chol_vector_a = 0.25 + torch.rand(*batch_shape, chol_param_number, dtype=ddtype)
chol_matrix_a = vector_to_triul(chol_vector_a)
means_a = torch.rand((*chol_vector_a.shape[:-1], event_shape), dtype=ddtype)

# Second Set of Parameters
chol_vector_b = 0.25+torch.rand(*batch_shape, chol_param_number, dtype=ddtype)
chol_matrix_b = vector_to_triul(chol_vector_b)
means_b = torch.rand((*chol_vector_b.shape[:-1], event_shape), dtype=ddtype)

# Flexible MultiVariateNormal
mvn_a_flexible = FlexibleMultivariateNormal(means_a, chol_matrix_a,
                                            init_natural=False, use_suff_stat_mean=True,
                                            use_suff_stat_variance=True, use_sample=True)
mvn_b_flexible = FlexibleMultivariateNormal(means_b, chol_matrix_b,
                                            init_natural=False, use_suff_stat_mean=True,
                                            use_suff_stat_variance=True, use_sample=True)

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
                                            init_natural=True, use_sample=True,
                                            use_suff_stat_mean=True, use_suff_stat_variance=True)
means_a_from_natural = mvn_a_naturals.loc
chol_matrix_a_from_natural = mvn_a_naturals.scale_tril

dmeans = torch.sum(torch.abs((means_a_from_natural - means_a)))
dchol_matrix = torch.sum(torch.abs((chol_matrix_a_from_natural - chol_matrix_a)))

mean0 = mvn_a_naturals.suff_stat_mean[0]
mmT = torch.linalg.matmul(mean0.unsqueeze(-1), mean0.unsqueeze(-2))
Sigma0 = mvn_a_naturals.suff_stat_mean[1] - mmT
Sigma1 = mvn_a_naturals.suff_stat_variance[0]

dchol_covariance = torch.sum(torch.abs(torch.linalg.cholesky(Sigma1) - chol_matrix_a))
dcovariance = torch.sum(torch.abs((mvn_a_standard.covariance_matrix - Sigma1)))

if (dmeans < 1e-6) & (dchol_matrix < 1e-6) & (dchol_matrix < 1e-6):
    print("Passed: Natural to Param and Param to Natural")
else:
    print('Failed test')

# Check Variance of the sufficient statistic
fourth_moment_flexible = mvn_a_naturals.suff_stat_variance[1]
fourth_moment_standard = torch.zeros(fourth_moment_flexible.shape, dtype=fourth_moment_flexible.dtype)
for ii in range(event_shape):
    for jj in range(event_shape):
        for kk in range(event_shape):
            for ll in range(event_shape):

                xx = jj + ii * event_shape
                yy = ll + kk * event_shape

                ss = Sigma1[..., ii, jj] * Sigma1[..., kk, ll] \
                     + Sigma1[..., ii, kk] * Sigma1[..., jj, ll] \
                     + Sigma1[..., ii, ll] * Sigma1[..., kk, jj]

                fourth_moment_standard[..., xx, yy] = ss
dfourthmoment = torch.sum(torch.abs((fourth_moment_flexible - fourth_moment_standard)))

if (dfourthmoment < 1e-6) :
    print("Passed: Fourth Centered Moment")
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
                                            init_natural=False, init_chol=False, use_suff_stat_mean=True,
                                            use_suff_stat_variance=True, use_sample=True)
mvn_a_naturals_mat = FlexibleMultivariateNormal(mvn_a_flexible.natural1,
                                            mvn_a_flexible.natural2,
                                            init_natural=True, init_chol=False, use_sample=True,
                                            use_suff_stat_mean=True, use_suff_stat_variance=True)

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
chol_matrix_c = vector_to_triul(chol_vector_c)
means_c = torch.rand((*chol_vector_c.shape[:-1], event_shape), dtype=ddtype)

# Flexible MultiVariateNormal
mvn_c_flexible = FlexibleMultivariateNormal(means_c, chol_matrix_c,
                                            init_natural=False, use_suff_stat_mean=True,
                                            use_suff_stat_variance=True, use_sample=True)

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
                                            init_natural=False, init_chol=True, use_suff_stat_mean=True)
                mvn_c_cur = FlexibleMultivariateNormal(means_c_cur, chol_matrix_c_cur,
                                            init_natural=False, init_chol=True, use_suff_stat_mean=False)
                non_broadcasted_kl[ii, jj, kk, ll] = flexible_kl(mvn_a_cur, mvn_c_cur)

if torch.sum(torch.abs(non_broadcasted_kl - broadcasted_kl)) < 1e-4:
    print('Passed: KL broadcasting')
else:
    print('Failed test')




