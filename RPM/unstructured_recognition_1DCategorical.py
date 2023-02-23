import torch
import numpy as np
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from utils import categorical_cross_entropy


class UnstructuredRecognition:
    """
    Recognition-Parametrized Model (RPM) for Categorical Peer-Supervision (Applied to MNIST)
        num_digits:          size of the categorical distribution
        observations:        tuple of conditionally independent observations x_i (num_obs x 28 x 28 for MNIST)
        recognition_network: conv.nn that outputs image identities log-probabilities (f_theta(z|x_i))
        factor_prior:        prior over images identity (uniform)
        factor_indpt:        output of the recognition network on observation
        latent:              posterior variational distribution over observations identity
    """

    def __init__(self, num_digits, observations, fit_params=None,
                 recognition_network=None, factor_prior=None, factor_indpt=None, latent=None):

        self.num_digits = num_digits
        self.fit_params = fit_params
        self.num_independent_factors = len(observations)
        self.num_obs = observations[0].shape[0]
        self.loss_tot = None

        # Init Variational Distribution q_rho(Z)
        if latent is None:
            self.init_variational()
        else:
            self.latent = latent

        # Init Factor Prior
        if factor_prior is None:
            self.init_prior()
        else:
            self.factor_prior = factor_prior

        # Init Recognition Network
        if recognition_network is None:
            self.recognition_network = Net(num_digits)
        else:
            self.recognition_network = recognition_network

        # Init / Update factors
        if factor_indpt is None:
            self.update_factors(observations)
        else:
            self.factor_indpt = factor_indpt

    def get_loss(self):

        # Model
        prior = self.factor_prior
        factors = self.factor_indpt
        variational = self.latent

        # Dimensions of the problem
        num_obs = self.num_obs
        num_factors = len(factors)

        # <log q>_q
        variational_entropy = - categorical_cross_entropy(variational, variational)

        # <log p>_q
        prior_xentropy = categorical_cross_entropy(variational, prior)

        # <log f(.|x)>_q
        factors_xentropy = \
            sum([categorical_cross_entropy(variational, factor_theta_i) for factor_theta_i in factors])

        # log p0
        loss_offset = num_obs * num_factors * torch.log(1 / torch.tensor([num_obs]))

        # < log \sum f(.|x)  >_q
        log_denominators = [torch.log(torch.mean(f_theta_i.probs, 0)) for f_theta_i in factors]
        denominators = [Categorical(logits=log_i.unsqueeze(dim=0).repeat(num_obs, 1)) for log_i in log_denominators]
        denominator_xentropy = sum([categorical_cross_entropy(variational, f_i) for f_i in denominators])

        # Free Energy / ELBO
        free_energy = \
            torch.sum(variational_entropy + prior_xentropy + factors_xentropy + loss_offset - denominator_xentropy)

        return - free_energy

    def fit(self, observations):
        # Fit parameters
        fit_params = self.fit_params
        ite_max = fit_params['ite_max']

        # Network Optimizer  Parameters
        learning_rate = 1 * 1e-7
        optimizer = torch.optim.SGD(self.recognition_network.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.1, patience=75, threshold=0.0001,
                                                               threshold_mode='abs')

        # Init Loss
        loss_tot = np.zeros(ite_max)

        for ite_em in range(ite_max):

            # E-Step
            self.update_variational()

            # Loss
            loss = self.get_loss()

            # Clear gradients for this training step
            optimizer.zero_grad()

            # Backpropagation, compute gradients
            loss.backward(retain_graph=True)

            # M-Step apply gradients
            optimizer.step()

            # Factors distributions are the output of a shared conv-net
            self.update_factors(observations)

            loss_tot[ite_em] = loss
            print('Iteration :' + str(ite_em + 1) + '/' + str(ite_max) + ' Loss ' + str(loss.detach().numpy()))
            scheduler.step(loss)

        self.loss_tot = loss_tot

    def update_variational(self):

        # Current Model
        num_obs = self.num_obs
        prior = self.factor_prior
        factors = self.factor_indpt

        # Approximate marginals
        log_denominators = sum([torch.log(torch.mean(f_theta_i.probs, 0)) for f_theta_i in factors])
        log_denominators = (log_denominators.unsqueeze(dim=0)).repeat(num_obs, 1)

        # Prior
        log_prior = prior.logits
        log_factor = sum([f_theta_i.logits for f_theta_i in factors])

        # log-prob of the variational
        logits_posterior = log_prior + log_factor - log_denominators

        self.latent = Categorical(logits=logits_posterior)

    def update_factors(self, observations):
        # Recognition network
        net = self.recognition_network

        # Factors NN Output
        factors_output_tot = [net.forward(x_i.unsqueeze(dim=1)) for x_i in observations]

        # Factors Distribution
        self.factor_indpt = [Categorical(logits=logitsc) for logitsc in factors_output_tot]

    def init_prior(self):
        num_obs = self.num_obs
        num_digits = self.num_digits

        logits_prior = torch.log(torch.ones(num_obs, num_digits) / num_digits)
        logits_prior.requires_grad = True
        self.factor_prior = Categorical(logits=logits_prior)

    def init_variational(self):
        num_obs = self.num_obs
        num_digits = self.num_digits

        logits_variational = torch.log(torch.ones(num_obs, num_digits) / num_digits)
        self.latent = Categorical(logits=logits_variational)

    def permute_prediction(self, predictions, labels, used_digits):
        # Find Best Prediction in case the train labels are permuted

        # Number Of used Digits
        num_digits = self.num_digits

        # Score Used to find Opt. Permutation
        scor_tot = torch.eye(num_digits)

        for digit_id in range(num_digits):
            digit_cur = used_digits[digit_id]

            # Prediction digit = digit_cur
            idx = (predictions == digit_cur)

            # How many times 'digit_cur' from prediction is mapped to each digit in label.
            scor_tot[digit_id, :] = (labels[idx].unsqueeze(dim=0) == used_digits.unsqueeze(dim=1)).sum(1)

        # Maximise scor_tot using Hungarian Algorithm
        _, perm = linear_sum_assignment(-scor_tot)

        return torch.tensor(perm)


class Net(nn.Module):
    # Convolutional Neural Network shared across independent factors
    def __init__(self, num_digits):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(4*4*20, 50)
        self.fc2 = nn.Linear(50, num_digits)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 4*4*20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


