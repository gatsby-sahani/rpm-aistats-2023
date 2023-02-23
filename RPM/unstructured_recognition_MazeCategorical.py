import torch
import numpy as np
import torch.nn as nn


import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from utils import categorical_cross_entropy, categorical_cross_entropy_2D
from scipy.optimize import linear_sum_assignment


class UnstructuredRecognition:
    """
    Recognition-Parametrized Model (RPM) for Maze Like Categorical Environment Latents Based on a HMM (RPM-HMM-Maze)
        num_state:           number of state in the latent space
        observations:        image trajectory tensor (num_obs x pixel1 x pixel2) - conditionally independent observations  
        recognition_network: conv.nn that outputs latent state identity log-probabilities (f_theta(z|x_i))
        factor_prior:        prior over images identity (uniform)
        transition_matrix:   latent state transitions
        factor_indpt:        output of the recognition network on observation
        latent:              posterior variational distribution over observations identity
    """

    def __init__(self, num_state, observations, fit_params=None,
                 recognition_network=None, factor_prior=None, transition_matrix=None, factor_indpt=None, latent=None):

        self.num_state = num_state
        self.fit_params = fit_params
        self.num_obs = observations[0].shape[0]
        self.loss_tot = None

        # Init Variational Distribution q_rho(Z)
        self.latent = latent

        # Init Factor Prior
        if factor_prior is None:
            self.init_prior()
        else:
            self.factor_prior = factor_prior

        # Init Factor Transition Matrix
        if factor_prior is None:
            self.init_transition_matrix()
        else:
            self.transition_matrix = transition_matrix

        # Init Recognition Network
        if recognition_network is None:
            self.recognition_network = Net(num_state)
        else:
            self.recognition_network = recognition_network

        # Init / Update factors
        if factor_indpt is None:
            self.update_factors(observations)
        else:
            self.factor_indpt = factor_indpt

    def fit(self, observations):
        # Fit parameters
        num_state = self.num_state
        fit_params = self.fit_params
        ite_max = fit_params['ite_max']
        recognition_network = self.recognition_network

        # Network Optimizer  Parameters
        if 'lr' in fit_params.keys():
            learning_rate = fit_params['lr']
        else:
            learning_rate = 5 * 1e-4

        # Perturb Transition Matrix Update
        if 'lambda' in fit_params.keys():
            lambda_tot = fit_params['lambda']
        else:
            lambda_tot = torch.linspace(0.6, 1, steps=ite_max)

        # Use Annealing
        if 'beta' in fit_params.keys():
            beta_tot = fit_params['beta']
        else:
            beta_tot = torch.linspace(1, 1, steps=ite_max)

        # Set Up optimizer
        optimizer = torch.optim.Adamax(recognition_network.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1, total_iters=ite_max)

        # Init Loss
        loss_tot = np.zeros(ite_max)

        #  torch.autograd.set_detect_anomaly(True)
        for ite_em in range(ite_max):

            with torch.no_grad():
                # E-Step for the latents
                self.update_variational(observations)

                # M-Step for HMM params (only valid for one chain !)
                self.update_transition_matrix()

                # Try to prevent the collapsing of some states by perturbing the transition matrix
                self.perturb_transition_matrix_update(lambda0=lambda_tot[ite_em])

            # Get Loss for M-Step on recognition network
            loss = self.get_loss(beta=beta_tot[ite_em])

            # Clear gradients for this training step
            optimizer.zero_grad()

            # Backpropagation, compute gradients
            loss.backward(retain_graph=True)

            # M-Step apply gradients
            optimizer.step()
            scheduler.step()

            # Recognition factors pass f_theta(z_t | x_t) (initial step uses log prob !)
            self.update_factors(observations)

            loss_tot[ite_em] = loss
            print('Iteration :' + str(ite_em + 1) + '/' + str(ite_max) + ' Loss ' + str(loss.detach().numpy()))

        self.loss_tot = loss_tot
        return loss_tot

    def get_loss(self,  beta=1):

        # Current Model
        f_theta_t = self.factor_indpt
        factor_prior_probs = self.factor_prior
        transition_matrix = self.transition_matrix
        latent_marginals, latent_pairwise_marginals = self.latent

        trajectory_length = latent_marginals.shape[0]

        # <log q>_q
        q_marginals = Categorical(probs=latent_marginals)
        variational_xentropy_singletons = categorical_cross_entropy(q_marginals, q_marginals)
        variational_xentropy_pairwise = categorical_cross_entropy_2D(latent_pairwise_marginals,
                                                                     latent_pairwise_marginals)
        variational_entropy = - torch.sum(variational_xentropy_pairwise) + torch.sum(
            variational_xentropy_singletons[1:-1])

        # <log p>_q
        # gamma[t,i] p(z[t]=i |x[1:T])
        # epsilon[t,i,j] = p(z[t]=i, z[t+1]=j | x[1:T])
        factor_prior = Categorical(probs=factor_prior_probs)
        variat_prior = Categorical(probs=latent_marginals[0])
        prior_xentropy_0 = categorical_cross_entropy(variat_prior, factor_prior)
        prior_xentropy_t = categorical_cross_entropy_2D(latent_pairwise_marginals, transition_matrix)
        prior_xentropy = prior_xentropy_0 + torch.sum(prior_xentropy_t)

        # <log f(.|x)>_q
        factors_xentropy = torch.sum(categorical_cross_entropy(q_marginals, f_theta_t))

        # < log \sum f(.|x)  >_q
        denominators = Categorical(
            probs=(1 / trajectory_length) * torch.sum(f_theta_t.probs, dim=0, keepdim=True).repeat(
                (trajectory_length, 1)))
        denominator_xentropy = torch.sum(categorical_cross_entropy(q_marginals, denominators))

        # Free Energy (ELBO)
        free_energy = beta * variational_entropy + prior_xentropy + factors_xentropy - denominator_xentropy

        return - free_energy

    def init_prior(self):
        # Prior
        num_state = self.num_state
        factor_prior_probs = torch.ones(num_state) / num_state
        factor_prior_probs.require_grad = False  # We can update it in closed form
        self.factor_prior = factor_prior_probs

    def init_transition_matrix(self):
        # Transition Matrix Init
        num_state = self.num_state
        #  transition_matrix = torch.rand(num_state, num_state)
        transition_matrix = torch.ones(num_state, num_state) / num_state
        transition_matrix = transition_matrix / torch.sum(transition_matrix, dim=1, keepdim=True)
        transition_matrix.require_grad = False  # We can update it in closed form
        self.transition_matrix = transition_matrix

    def update_factors(self, observations):
        # Recognition factors pass f_theta(z_t | x_t) (initial step uses log prob !)
        recognition_network = self.recognition_network
        f_theta_t_tmp = recognition_network.forward(observations.unsqueeze(1))
        f_theta_t = Categorical(logits=f_theta_t_tmp)
        self.factor_indpt = f_theta_t

    def update_variational(self, observations):
        #  latent_marginals, latent_pairwise_marginals
        self.latent = self.recognition_baum_welch(observations)

    def update_transition_matrix(self):
        # M-Step for HMM params (only valid for one chain !)
        latent_marginals, latent_pairwise_marginals = self.latent
        transition_matrix = torch.sum(latent_pairwise_marginals, dim=0) / torch.sum(latent_marginals[:-1],
                                                                                    dim=0).unsqueeze(1)
        self.transition_matrix = transition_matrix

    def recognition_baum_welch(self, observations):
        # Baum-Welch Forward-Backward pass to estimate the marginals and pairwise marginals
        # from p(Z[1:T]|X[1:T])
        # gamma[t,i] p(z[t]=i |x[1:T])
        # epsilon[t,i,j] = p(z[t]=i, z[t+1]=j | x[1:T])

        f_theta_t = self.factor_indpt
        factor_prior_probs = self.factor_prior
        transition_matrix = self.transition_matrix
        num_state = self.num_state

        # Length of Current Trajectory
        trajectory_length = observations.shape[0]

        # Denominator of Unsupervised recognition model
        denominator = torch.sum(f_theta_t.probs, dim=0, keepdim=True) / trajectory_length

        # p_t in Baulm-Whelch algorithm
        emission_probs = f_theta_t.probs / denominator
        emission_probs = emission_probs / torch.sum(emission_probs, dim=1, keepdim=True)

        # Init VoI
        norm_tot = torch.zeros(trajectory_length)
        beta_tot = torch.zeros(trajectory_length, num_state)
        alpha_tot = torch.zeros(trajectory_length, num_state)
        epsilon_tot = torch.zeros(trajectory_length - 1, num_state, num_state)

        # Initial Step(s)
        alpha_tot[0] = factor_prior_probs * emission_probs[0]
        norm_tot[0] = torch.sum(alpha_tot[0])
        alpha_tot[0] /= norm_tot[0]
        beta_tot[-1] = 1

        # Forward Pass
        for t in range(trajectory_length - 1):
            alpha_tot[t + 1] = torch.linalg.matmul(transition_matrix.transpose(0, 1), alpha_tot[t]) * emission_probs[
                t + 1]
            norm_tot[t + 1] = torch.sum(alpha_tot[t + 1])
            alpha_tot[t + 1] /= norm_tot[t + 1]

        # Backward Pass
        for t in range(trajectory_length - 1 - 1, -1, -1):
            beta_tot[t] = torch.linalg.matmul(transition_matrix, (beta_tot[t + 1] * emission_probs[t + 1])) / norm_tot[
                t + 1]

        # Marginals Pairwise marginals
        gamma_tot = alpha_tot * beta_tot

        # Pairwise marginals
        for t in range(trajectory_length - 1):
            epsilon_tot[t] = \
                transition_matrix * \
                torch.linalg.matmul(alpha_tot[t].unsqueeze(1), (beta_tot[t + 1] * emission_probs[t + 1]).unsqueeze(0)) \
                / norm_tot[t + 1]

        return gamma_tot, epsilon_tot

    def perturb_transition_matrix_update(self, lambda0=1):
        # Try to prevent the collapsing of some states by perturbing the transition matrix

        num_state = self.num_state
        transition_matrix = self.transition_matrix
        perturbation = torch.ones(num_state, num_state) / num_state

        self.transition_matrix = lambda0 * transition_matrix + (1 - lambda0) * perturbation

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


def permute_prediction(predictions, labels, used_digits):
    # Find Best Prediction in case the train labels are permuted

    # Number Of used Digits
    num_digits = len(used_digits)

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


class MazeNet(nn.Module):
    def __init__(self, num_states, visual_field):
        self.visual_field = visual_field
        super(MazeNet, self).__init__()
        self.fc1 = nn.Linear(visual_field*visual_field, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_states)

    def forward(self, x):
        x = x.view(-1, self.visual_field*self.visual_field)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1)



