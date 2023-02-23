import torch
import numpy as np
import matplotlib.pyplot as plt


def compare_dist(posterior_true, q):

    num_samples, dim_latent = posterior_true.loc.shape

    mean_true = posterior_true.loc.detach().numpy()
    mean_fit = q.loc.detach().detach().numpy()

    var_true = posterior_true.covariance_matrix.detach().numpy()
    var_fit = q.covariance_matrix.detach().numpy()

    plt.figure(figsize=(5, 4*dim_latent))

    for dim_cur in range(dim_latent):

        mean_true_cur = mean_true[:, dim_cur]
        mean_fit_cur = mean_fit[:, dim_cur]

        plt.subplot(dim_latent+1, 1, dim_cur + 1)
        plt.plot([min(mean_true_cur), max(mean_true_cur)], [min(mean_true_cur), max(mean_true_cur)], color='k')
        plt.scatter(mean_true_cur, mean_fit_cur)

        if dim_cur == 0:
            plt.title('Posterior Fit. Var True/Fit')
        plt.xlabel('P(Z|X) MAP')
        plt.ylabel('q(Z) MAP')

    plt.subplot(dim_latent + 1, 1, dim_latent + 1 )
    plt.scatter(np.arange(dim_latent**2), var_true[0, :, :].flatten(), color='k')
    plt.scatter(np.arange(dim_latent**2), var_fit[0, :, :].flatten())


def compare_factors(zz, Xobs, posterior_true_pdf, posterior_factors):
    XX1 = 10
    XX2 = 0

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(zz.squeeze().numpy(), posterior_true_pdf[:, XX1], color='k', label='True')
    plt.plot(zz.squeeze().numpy(), posterior_factors[:, XX1], label='factors')
    plt.title('P(Z| X_1)')
    plt.xlabel('Z')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(zz.squeeze().numpy(), posterior_true_pdf[:, XX2], color='k')
    plt.plot(zz.squeeze().numpy(), posterior_factors[:, XX2])
    plt.title('P(Z| X_2)')
    plt.xlabel('Z')

    plt.subplot(2, 2, 2)
    plt.imshow(posterior_factors,
               extent=[Xobs.min().numpy(), Xobs.max().numpy(), zz.min().numpy(), zz.max().numpy()],
               aspect='auto')

    plt.ylabel('Z')
    plt.xlabel('X')
    plt.title('P(Z|X)')

    plt.subplot(2, 2, 4)
    plt.imshow(posterior_true_pdf,
               extent=[Xobs.min().numpy(), Xobs.max().numpy(), zz.min().numpy(), zz.max().numpy()],
               aspect='auto')

    plt.ylabel('Z')
    plt.xlabel('X')
    plt.title('P(Z|X) True')
