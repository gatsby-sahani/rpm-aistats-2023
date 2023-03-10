import torch
import numpy as np
import torch.nn as nn

import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.dirichlet import Dirichlet
from utils import categorical_cross_entropy


class UnstructuredRecognition:
    """
    Recognition-Parametrized Model (RPM) for Latent Dirichlet Allocation (LDA)
        num_texture:         set number of texture group
        observations:        list containing num_obs x dim_obs x dim_obs conditionally independent observed patches
        recognition_network: conv.nn that outputs patch textural probabilities
        factor_prior:        alpha parameter of Dirichlet distribution
        factor_indpt:        output of the recognition network on observation
        latent:              q(Z, theta) Variational Distribution over texture distribution and patch latent texture
    """

    def __init__(self, num_texture, observations, fit_params=None,
                 recognition_network=None, factor_prior=None, factor_indpt=None, latent=None):

        self.num_texture = num_texture
        self.fit_params = fit_params
        self.num_independent_factors = len(observations)
        self.num_obs = observations[0].shape[0]
        self.loss_tot = None

        # Init Factor Prior
        if factor_prior is None:
            self.init_prior()
        else:
            self.factor_prior = factor_prior

        # Init Variational Distribution q_rho(Z)
        if latent is None:
            self.init_variational()
        else:
            self.latent = latent

        # Init Recognition Network
        if recognition_network is None:
            self.recognition_network = Net(num_texture)
        else:
            self.recognition_network = recognition_network

        # Init / Update factors
        if factor_indpt is None:
            self.update_factors(observations)
        else:
            self.factor_indpt = factor_indpt

    def fit(self, observations):
        # Fit parameters
        fit_params = self.fit_params
        ite_max = fit_params['ite_max']

        # Network Optimizer  Parameters
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(self.recognition_network.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

        # Init Loss
        loss_tot = np.zeros(ite_max)

        for ite_em in range(ite_max):

            # E-Step
            with torch.no_grad():
                self.update_variational()

            # Loss
            self.update_factors(observations)

            loss = self.get_loss()

            # Clear gradients for this training step
            optimizer.zero_grad()

            # Backpropagation, compute gradients
            loss.backward()

            # M-Step apply gradients
            optimizer.step()

            # Store and print loss
            loss_tot[ite_em] = loss
            print('Iteration :' + str(ite_em + 1) + '/' + str(ite_max) + ' Loss ' + str(loss.detach().numpy()))
            scheduler.step(loss)

        self.loss_tot = loss_tot

    def get_loss(self):
        # - Free Energy (ELBO) of the RPM-LDA model

        # Variational
        (alpha_variational, gamma_variational) = self.latent

        # Factors
        factors = self.factor_indpt

        # Dimensions of the problem
        num_obs = self.num_obs
        num_factors = len(factors)

        # log p0
        loss_offset = num_obs * num_factors * torch.log(1 / torch.tensor([num_obs]))

        # H[q]
        alpha_tmp = Dirichlet(alpha_variational)
        variational_entropy_alpha = alpha_tmp.entropy().sum()
        variational_entropy_gamma = gamma_variational.entropy().sum()
        variational_entropy = (variational_entropy_alpha + variational_entropy_gamma)

        # <log p>_q
        alpha_qtot = alpha_variational.sum(-1)
        prior_xentropy = -variational_entropy_alpha - (torch.lgamma(alpha_qtot) - torch.lgamma(alpha_variational).sum(-1) ).sum()

        # <log f(.|x)>_q
        log_factors_all = torch.cat([fj.logits.unsqueeze(1) for fj in factors], dim=1)
        factors_xentropy = categorical_cross_entropy(gamma_variational, Categorical(logits=log_factors_all)).sum()

        # < log \sum f(.|x)  >_q
        log_denominators = torch.cat([torch.log(torch.mean(f_theta_i.probs, 0)).unsqueeze(0) for f_theta_i in factors],dim=0)\
            .unsqueeze(0).repeat(num_obs, 1, 1)
        denominator_xentropy = \
            categorical_cross_entropy(gamma_variational, Categorical(logits=log_denominators)).sum()

        # Free Energy / ELBO
        free_energy = loss_offset + variational_entropy + prior_xentropy + (factors_xentropy - denominator_xentropy)

        return - free_energy

    def update_variational(self):

        alpha0 = self.factor_prior[0]
        factors = self.factor_indpt
        current_variational = self.latent

        # Update q(Z)
        digamma = torch.digamma(current_variational[0]).unsqueeze(1)

        # Factors
        log_factor = torch.cat([f_theta_i.logits.unsqueeze(1) for f_theta_i in factors], dim=1)

        # Mixture
        log_denominators = torch.cat(
            [torch.log(torch.mean(f_theta_i.probs, 0)).unsqueeze(0).unsqueeze(0) for f_theta_i in factors]
            , dim=1)

        # Variational Update1
        logits_posterior = digamma + log_factor - log_denominators
        gamma_posterior = Categorical(logits=logits_posterior)

        # Update alpha
        alpha_posterior = alpha0 + gamma_posterior.probs.sum(dim=1)

        # Update variationals
        self.latent = (alpha_posterior, gamma_posterior)

    def update_factors(self, observations):
        # Recognition network
        net = self.recognition_network

        # Factors NN Output
        factors_output_tot = [net.forward(x_i.unsqueeze(dim=1)) for x_i in observations]

        # Factors Distribution
        self.factor_indpt = [Categorical(logits=logitsc) for logitsc in factors_output_tot]

    def init_prior(self):
        # Prior consist in P(z , theta)
        num_obs = self.num_obs
        num_texture = self.num_texture

        # Dirichlet Prior Param
        alpha = torch.tensor([0.1])

        # Uniform P(z=k | theta_n) thet_k_n
        # (not used)
        theta = torch.ones(num_obs, num_texture) / num_texture
        thetap = Categorical(probs=theta)

        self.factor_prior = (alpha, thetap)

    def init_variational(self):
        alpha = self.factor_prior[0]
        num_obs = self.num_obs
        num_texture = self.num_texture
        num_independent_factors = self.num_independent_factors
        gamma_n_j_k = torch.ones(num_obs, num_independent_factors, num_texture) / num_texture
        alpha_n_k = alpha + torch.sum(gamma_n_j_k, dim=1)

        gamma = Categorical(probs=gamma_n_j_k)

        self.latent = (alpha_n_k, gamma)


class Net(nn.Module):
    # Convolutional Neural Network shared across independent factors
    def __init__(self, num_texture):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(4*4*20, 50)
        self.fc2 = nn.Linear(50, num_texture)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 4*4*20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


# def xentropy_dirichlet(self):
#     k = self.concentration.size(-1)
#     a0 = self.concentration.sum(-1)
#     return (torch.lgamma(self.concentration).sum(-1) - torch.lgamma(a0) -
#             (k - a0) * torch.digamma(a0) -
#             ((self.concentration - 1.0) * torch.digamma(self.concentration)).sum(-1))
#
#     def update_prior(self):
#         alpha = self.factor_prior[0]
#         gamma_varional_n_j_k = self.latent[1]
#
#         theta_n_k = alpha + gamma_varional_n_j_k.probs.sum(1)
#         theta_n_k = theta_n_k / (1e-20 + theta_n_k.sum(-1, keepdim=True))
#         self.factor_prior = (alpha, Categorical(probs=theta_n_k))