# Imports
import pickle
import numpy as np

import torch
from torch import log
from torch import matmul
from utils import gp_kernel, optimizer_wrapper, print_loss, diagonalize, soft_bound, threshold_eigh, fast_XDXT
from flexible_multivariate_normal import FlexibleMultivariateNormal, flexible_kl, vector_to_triul, vectorize, kl

from networks import Net


class UnstructuredRecognition:
    """
        Unstructured Recognition with RP-GPFA:
        Recognition Parametrised Gaussian Process Factor Analysis

        Args:
            observations (Tensor): size num_observation x len_observation x dim_observation. Input.
            observation_locations (Tensor): size len_observation x dim_locations. Location of the Observations
            inducing_locations (Tensor): size len_observation x dim_locations. Location of inducing points
            fit_params (Dict): Fit parameters (dim_latent, inference method, lr, optimizer, etc...)

            Optional Args for initialisation:
                inducing_points_param (Tuple)
                gp_prior_param (Tuple)
                recognition_function (nn.module)
                factors_tilde_params(Tuple)

        Note:
            The fitting procedure returns:
                factors: recognition distribution approximating p(z_t | x_t)
                inducing_points_param: mean and chol(variance) of the variational inducing points q(U)
                variational_marginals: q(z_t)
        """

    def __init__(self, observations, observation_locations,
                 inducing_locations=None, inducing_points_param=None,
                 gp_prior_param=None, recognition_function=None,
                 factors_tilde_params=None,
                 fit_params=None):

        # Transform Observation in length 1 tuple if necessary
        if not (type(observations)) is tuple:
            observations = (observations,)

        # Check that all factors dimensions are consistent
        num_factors = len(observations)
        num_observation, len_observation = observations[0].shape[:2]
        for cur_factor in range(num_factors - 1):
            cur_num_observation, cur_len_observation = observations[cur_factor + 1].shape[:2]
            assert cur_num_observation == num_observation
            assert cur_len_observation == len_observation

        # Device and data type
        self.dtype = observations[0].dtype
        self.device = observations[0].device
        print(self.device)

        # Dimension of the problem
        self.num_factors = num_factors
        self.num_observation = num_observation
        self.len_observation = len_observation

        # Fit and default parameters
        if fit_params is None:
            self.fit_params = {}
        else:
            self.fit_params = fit_params
        self.init_fit_params()
        self.dim_latent = fit_params['dim_latent']

        # Auxiliary Factors for 'VariationalBound' inference
        self.factors_tilde_params = None

        # Mean and Variance of the Inducing points
        self.inducing_points_param = None

        # Inference Mode
        if self.fit_params['inference_mode'] == '2ndOrder':
            self.marginals_use_suff_stat_variance = True
            self.marginals_use_samples = False
            self.factors_use_suff_stat_mean = False
        elif self.fit_params['inference_mode'] == 'Samples':
            self.marginals_use_suff_stat_variance = False
            self.marginals_use_samples = True
            self.factors_use_suff_stat_mean = False
        elif self.fit_params['inference_mode'] == 'MomentMatching':
            self.marginals_use_suff_stat_variance = False
            self.marginals_use_samples = False
            self.factors_use_suff_stat_mean = True
        elif self.fit_params['inference_mode'] == 'VariationalBound':
            self.marginals_use_suff_stat_variance = False
            self.marginals_use_samples = False
            self.factors_use_suff_stat_mean = False
        else:
            raise NotImplementedError()

        # The Recognition function(s) go from observation to factors parameters
        if recognition_function is None:
            if self.fit_params['constraint_factors'] == 'diag':
                full_covariance = False
            elif self.fit_params['constraint_factors'] == 'full':
                full_covariance = True
            else:
                raise NotImplementedError()

            self.recognition_function = ()
            for cur_factor in range(self.num_factors):
                cur_dim_hidden = self.fit_params['dim_hidden'][cur_factor]
                cur_nn_type = self.fit_params['nn_type'][cur_factor]

                cur_dim_latent = self.dim_latent
                cur_dim_observation = observations[cur_factor].shape[2:]

                if len(cur_dim_observation) == 1:
                    cur_dim_observation = cur_dim_observation[0]

                cur_recognition_function = Net(cur_dim_latent, cur_dim_observation,
                                               dim_hidden=cur_dim_hidden,
                                               full_covariance=full_covariance,
                                               nn_type=cur_nn_type
                                               ).to(self.device.index)
                self.recognition_function += (cur_recognition_function,)

        else:
            self.recognition_function = recognition_function

        # Use the recognition Function to update the factors
        self.factors = None
        self.update_factors(observations)

        # Set inducing locations
        if inducing_locations is None:
            raise ValueError('Must provide Inducing Locations')
        self.inducing_locations = inducing_locations
        self.observation_locations = observation_locations
        self.num_inducing_point = inducing_locations.shape[0]

        # Initialize Inducing Point distributions
        self.inducing_points = None
        if inducing_points_param is None:
            self.init_inducing_points()
        else:
            self.inducing_points_param = inducing_points_param
        self.update_inducing_points(init=True)

        # Initialize GP-prior parameters
        self.gp_kernel = gp_kernel(self.fit_params['gp_kernel'])
        if gp_prior_param is None:
            self.init_gp_prior_param()
        else:
            self.gp_prior_param = gp_prior_param

        # Update prior at inducing locations
        self.prior = None
        self.update_prior()

        # Update variational marginals distribution
        self.variational_marginals = None
        self.update_variational_marginals()

        # Auxiliary Factors Init for interior variational Bound
        if self.fit_params['inference_mode'] == 'VariationalBound':
            if factors_tilde_params is None:
                self.init_factors_tilde()
            else:
                self.factors_tilde_params = factors_tilde_params
            self.update_factors_tilde()
        else:
            self.factors_tilde = None
            self.factors_tilde_params = None

        # Moment Matched Denominators if necessary
        if self.fit_params['inference_mode'] == 'MomentMatching':
            self.moment_match_denominator()
        else:
            self.factors_ratio_moment_matched = None
            self.factors_ratio_log_beta = None

        self.loss_tot = None

    def fit(self, observations):
        """ Fit an RP-GPFA to observations """

        if not (type(observations)) is tuple:
            observations = (observations,)

        # Grasp Fit parameters
        fit_params = self.fit_params

        # Inner and outter iterations
        ite_out = fit_params['ite_out']

        # Optimizer Prio
        optimizer_prior = optimizer_wrapper([*self.gp_prior_param],
                                            fit_params['optimizer_prior']['name'],
                                            **fit_params['optimizer_prior']['param'])

        # Optimizer Factors
        factors_parameter = []
        for cur_factor in range(self.num_factors):
            factors_parameter += self.recognition_function[cur_factor].parameters()
        optimizer_factors = optimizer_wrapper(factors_parameter,
                                              fit_params['optimizer_factors']['name'],
                                              **fit_params['optimizer_factors']['param'])

        # Optimizer Inducing Points
        optimizer_inducing_points = optimizer_wrapper([*self.inducing_points_param],
                                                      fit_params['optimizer_inducing_points']['name'],
                                                      **fit_params['optimizer_inducing_points']['param'])
        # Reset Optimizer
        optimizer_prior.zero_grad()
        optimizer_factors.zero_grad()
        optimizer_inducing_points.zero_grad()

        # Optimizer Auxiliary factors
        if self.fit_params['inference_mode'] == 'VariationalBound':
            optimizer_factors_tilde = optimizer_wrapper(
                [factor_i for factor in self.factors_tilde_params for factor_i in factor],
                fit_params['optimizer_factors_tilde']['name'],
                **fit_params['optimizer_factors_tilde']['param'])
            optimizer_factors_tilde.zero_grad()

        # Init Loss
        loss_cur = self.get_loss()
        self.loss_tot = None
        loss_tot = np.array(loss_cur.cpu().detach().numpy()[0])
        print('Iterations ' + str(0) + '/' + str(ite_out) + ' Loss: %.6e' % loss_tot)

        for ite in range(ite_out):

            # Update GP Prior
            loss_tot = self.step_prior(observations, optimizer_prior, loss_tot)
            print_loss(1 + ite, ite_out, loss_tot[-1], sub_step='Prior')

            # Update Inducing Points
            loss_tot = self.step_inducing_points(observations, optimizer_inducing_points, loss_tot)
            print_loss(1 + ite, ite_out, loss_tot[-1], sub_step='Inducing Points')

            # Update Recognition Factors
            loss_tot = self.step_factors(observations, optimizer_factors, loss_tot)
            print_loss(1 + ite, ite_out, loss_tot[-1], sub_step='Factors')
            self.loss_tot = loss_tot

            # Update Auxiliary Factors
            if self.fit_params['inference_mode'] == 'VariationalBound':
                loss_tot = self.step_factors_tilde(observations, optimizer_factors_tilde, loss_tot)
                print_loss(1 + ite, ite_out, loss_tot[-1], sub_step='Factors Auxiliary')
                self.loss_tot = loss_tot

        return loss_tot

    def get_loss(self, use_minibatch=False):
        """ The loss is defined as the negative Free Energy """

        # Constant Term of the free energy
        num_factors = self.num_factors
        num_observation = self.num_observation
        len_observation = self.len_observation

        if not self.fit_params['ergodic']:
            constant = - len_observation * num_observation * num_factors \
                       * log(torch.tensor([num_observation], dtype=self.dtype, device=self.device))
        else:
            constant = - len_observation * num_observation * num_factors \
                       * log(torch.tensor([num_observation * len_observation], dtype=self.dtype, device=self.device))

        # KL[variational, prior] for Inducing points
        KLqUpU = flexible_kl(self.inducing_points, self.prior, repeat1=[0, 1], repeat2=[1])

        # KL with factors and denominator depends on inference mode
        if self.fit_params['inference_mode'] == '2ndOrder':
            free_energy_theta = self.get_free_energy_theta_approx()

        elif self.fit_params['inference_mode'] == 'Samples':
            free_energy_theta = self.get_free_energy_theta_samples(num_samples=self.fit_params['num_samples'])

        elif self.fit_params['inference_mode'] == 'MomentMatching':
            free_energy_theta = self.get_free_energy_theta_moment_match()

        elif self.fit_params['inference_mode'] == 'VariationalBound':
            free_energy_theta = self.get_free_energy_theta_bound()

        # Train recognition networks using only a subset of the observations
        if use_minibatch:
            kept_batch_n = np.random.choice(range(self.num_observation), int(0.5 * self.num_observation))
            kept_batch_t = np.random.choice(range(self.len_observation), int(0.75 * self.len_observation))
            KLqUpU = KLqUpU[kept_batch_n, :]
            free_energy_theta = free_energy_theta[kept_batch_n][:, kept_batch_t]
        free_energy = constant - KLqUpU.sum() + free_energy_theta.sum()

        return - free_energy

    def get_free_energy_theta_bound(self):
        """ Variational Lower Bound Of the Free Energy """

        # Dimension of the problem
        num_factors = self.num_factors
        num_observation = self.num_observation

        # Init across factors
        diag_delta_factors_log_normalizer = 0
        log_Gamma_t_n = 0
        KLqfhat = 0

        # Loop across conditionally independent factors
        for ii in range(num_factors):

            # Natural Parameters of the factors ~ 1 x N x T x K (x K)
            factors_natural1 = self.factors[ii].natural1.unsqueeze(0)
            factors_natural2 = self.factors[ii].natural2.unsqueeze(0)
            factors_log_normaliser = self.factors[ii].log_normalizer.unsqueeze(0)

            # Pseudo Natural Parameters of the auxiliary factors ~ N x 1 x T x K (x K)
            factors_tilde_natural1 = self.factors_tilde[ii][0].unsqueeze(1)
            factors_tilde_natural2 = self.factors_tilde[ii][1].unsqueeze(1)

            # eta_m - eta_tilde_n ~ N x N x T x K (x K)
            delta_natural1 = factors_natural1 - factors_tilde_natural1
            delta_natural2 = factors_natural2 - factors_tilde_natural2

            # Enforce factors ratio to be a valid distribution
            if self.fit_params['constraint_factors'] == 'diag':
                delta_natural2 = soft_bound(delta_natural2, bound=0, beta=10, mode='upper')
            elif self.fit_params['constraint_factors'] == 'full':
                delta_natural2 = threshold_eigh(delta_natural2, bound=-1e-6, beta=10, mode='upper')
            else:
                raise NotImplementedError()

            # Build All the ratio distribution with natural parameter eta_m - eta_tilde_n
            delta_factors = FlexibleMultivariateNormal(delta_natural1, delta_natural2,
                                                       init_natural=True, init_chol=False)

            # Grasp only the m = n distribution for KL estimation
            diag_delta_natural1 = delta_factors.natural1[range(num_observation), range(num_observation)]
            diag_delta_natural2 = delta_factors.natural2[range(num_observation), range(num_observation)]
            diag_delta_factors_log_normalizer_cur = \
                delta_factors.log_normalizer[range(num_observation), range(num_observation)]
            diag_delta_factors_log_normalizer += \
                diag_delta_factors_log_normalizer_cur - factors_log_normaliser.squeeze(0)

            # Natural Parameter from the marginal
            variational_natural1 = self.variational_marginals.natural1
            variational_natural2 = self.variational_marginals.natural2
            variational_suff_stat = self.variational_marginals.suff_stat_mean
            variational_log_normalizer = self.variational_marginals.log_normalizer

            # KL[q || f(.|x) / f_tilde]
            KLqfhat += kl(
                (variational_natural1, variational_natural2),
                (diag_delta_natural1, diag_delta_natural2),
                variational_log_normalizer, diag_delta_factors_log_normalizer_cur,
                variational_suff_stat)

            # Normaliser f(.) / f_tilde
            log_Gamma_t_n += torch.logsumexp(delta_factors.log_normalizer - factors_log_normaliser, dim=1) \
                             - log(torch.tensor([num_observation], dtype=self.dtype, device=self.device))

        return diag_delta_factors_log_normalizer - log_Gamma_t_n - KLqfhat

    def get_free_energy_theta_samples(self, num_samples=30):
        """ Estimate Free Energy using samples from the variational marginals """

        # Dimension of the problem
        num_factors = self.num_factors
        num_observations = self.num_observation
        len_observations = self.len_observation

        # Entropy[Variational]
        Hq = num_factors * self.variational_marginals.entropy

        # Sample from Variational marginals
        ergodic = self.fit_params['ergodic']
        if not ergodic:
            constant = log(torch.tensor([num_observations], dtype=self.dtype, device=self.device))
            samples = self.variational_marginals.rsample(torch.Size([num_samples])).unsqueeze(-3)
        else:
            constant = log(torch.tensor([num_observations * len_observations], dtype=self.dtype, device=self.device))
            samples = self.variational_marginals.rsample(torch.Size([num_samples])).unsqueeze(-2).unsqueeze(-2)

        # Init
        KLqf = 0
        log_prob_s_n_t = 0

        # Loop over conditionally independent factors
        for cur_factor in self.factors:

            # KL[ variational || recognition factor]
            KLqf += flexible_kl(self.variational_marginals, cur_factor)

            # Estimate denominator with samples
            log_prob_s_n_m_t = cur_factor.log_prob(samples)

            if not ergodic:
                log_prob_s_n_t += (torch.logsumexp(log_prob_s_n_m_t, dim=-2) - constant).mean(dim=0)
            else:
                # log_prob_s_n_t += (torch.logsumexp(log_prob_s_n_m_t, dim=(-2, -3)) - constant).mean(dim=0)
                log_prob_s_n_t += (torch.logsumexp(log_prob_s_n_m_t, dim=(-1, -2)) - constant).mean(dim=0)

        return - KLqf - Hq - log_prob_s_n_t

    def get_free_energy_theta_approx(self):
        """ Estimate Free Energy using a second order approximation for Eq(log f) """

        # Dimension of the problem
        dim_latent = self.dim_latent
        num_factors = self.num_factors
        num_observations = self.num_observation
        len_observations = self.len_observation

        # Init Fjt
        log_denominator = 0
        KLqf = 0

        # Entropy[Variational]
        Hq = num_factors * self.variational_marginals.entropy

        ergodic = self.fit_params['ergodic']
        if not (ergodic):
            # Eq(T(Z)) ~ N x 1 x T x K (x K)
            suff_stat_q_mean1 = self.variational_marginals.suff_stat_mean[0].unsqueeze(1)
            suff_stat_q_mean2 = self.variational_marginals.suff_stat_mean[1].unsqueeze(1)

            # Vq(T(Z)) ~ N x 1 x 1 x T x K(**2) x K(**2)
            suff_stat_q_variance1 = self.variational_marginals.suff_stat_variance[0].unsqueeze(1).unsqueeze(1)
            suff_stat_q_variance2 = self.variational_marginals.suff_stat_variance[1].unsqueeze(1).unsqueeze(1)

            # Loop over factors
            for cur_factor in self.factors:
                # KL[ variational || recognition factor]
                KLqf += flexible_kl(self.variational_marginals, cur_factor)

                # Natural Parameters of the factors ~ 1 x N x T x K (x K)
                factors_natural1 = cur_factor.natural1.unsqueeze(0)
                factors_natural2 = cur_factor.natural2.unsqueeze(0)
                factors_log_normaliser = cur_factor.log_normalizer.unsqueeze(0)

                # Extend q: N x 1 x T x K (x K) and f: 1 x N x T x K (x K)
                term1 = matmul(suff_stat_q_mean1.unsqueeze(-2), factors_natural1.unsqueeze(-1)).squeeze(-1).squeeze(-1)
                term2 = (suff_stat_q_mean2 * factors_natural2).sum(dim=(-1, -2))
                log_prob_n_m_t = term1 + term2 - factors_log_normaliser

                # log f(E(T(Z))) ~ N x 1 x T
                log_sum_exp_n_t = torch.logsumexp(log_prob_n_m_t, dim=1, keepdim=True)
                log_denominator_suff_stat = (log_sum_exp_n_t - log(torch.tensor([num_observations],
                                                                                dtype=self.dtype,
                                                                                device=self.device))).squeeze(1)

                # Weights ~ N x T x N should normalize across last dimension
                pi_n_m_t = torch.exp(log_prob_n_m_t - log_sum_exp_n_t).permute((0, 2, 1))
                if not (torch.abs(pi_n_m_t.sum(-1) - 1).max() < 5e-3):
                    raise ValueError('Incorect Shape normalization for Pi')

                # diag(Weights) ~ N x T x N x N
                diag_pi_n_t_M = torch.zeros(pi_n_m_t.shape + torch.Size([self.num_observation]),
                                            dtype=self.dtype, device=self.device)
                diag_pi_n_t_M[..., range(self.num_observation), range(self.num_observation)] = pi_n_m_t

                # diag(Weights) + Weights.Weights.T  ~ N x T x N x N
                PI_n_t_m1_m2 = diag_pi_n_t_M - matmul(pi_n_m_t.unsqueeze(-1), pi_n_m_t.unsqueeze(-2))

                # First term eta.T V eta ~ N x N x N x T (K x K matrix products)
                etaVeta1 = matmul(
                    factors_natural1.unsqueeze(1).unsqueeze(-2),
                    matmul(suff_stat_q_variance1,
                           factors_natural1.unsqueeze(2).unsqueeze(-1))
                ).squeeze(-1).squeeze(-1)

                # Second term eta.T V eta ~ N x N x N x T (K**2 x K**2 matrix products)
                etaVeta2 = matmul(
                    vectorize(factors_natural2).unsqueeze(1).unsqueeze(-2),
                    matmul(suff_stat_q_variance2,
                           vectorize(factors_natural2).unsqueeze(2).unsqueeze(-1))
                ).squeeze(-1).squeeze(-1)

                # Reshape eta.T V eta ~ N x T x N x N
                etaVeta_n_t_m1_m2 = (etaVeta1 + etaVeta2).permute((0, 3, 1, 2))

                # Second order Approximation for Eq(log f)
                log_denominator += log_denominator_suff_stat + 0.5 * (etaVeta_n_t_m1_m2 * PI_n_t_m1_m2).sum((-1, -2))

        else:
            # Eq(T(Z)) ~ N x 1 x T x 1 x K (x K)
            suff_stat_q_mean1 = self.variational_marginals.suff_stat_mean[0].unsqueeze(1).unsqueeze(3)
            suff_stat_q_mean2 = self.variational_marginals.suff_stat_mean[1].unsqueeze(1).unsqueeze(3)

            # Vq(T(Z)) ~ N x 1 x 1 x T x K(**2) x K(**2)
            suff_stat_q_variance1 = self.variational_marginals.suff_stat_variance[0]
            suff_stat_q_variance2 = self.variational_marginals.suff_stat_variance[1]

            # Loop over factors
            for cur_factor in self.factors:
                # KL[ variational || recognition factor]
                KLqf += flexible_kl(self.variational_marginals, cur_factor)

                # Natural Parameters of the factors ~ 1 x N' x 1 x T x K (x K)
                factors_natural1 = cur_factor.natural1.unsqueeze(0).unsqueeze(2)
                factors_natural2 = cur_factor.natural2.unsqueeze(0).unsqueeze(2)
                factors_log_normaliser = cur_factor.log_normalizer.unsqueeze(0).unsqueeze(2)

                # Extend q: N x 1  x T x 1  x K and f: 1 x N' x 1 x T' x K
                term1 = matmul(suff_stat_q_mean1.unsqueeze(-2),
                               factors_natural1.unsqueeze(-1)).squeeze(-1).squeeze(-1)
                term2 = (suff_stat_q_mean2 * factors_natural2).sum(dim=(-1, -2))

                # Mixture evaluated at mean ~ N x N' x T x T'
                log_prob_n_m_t_r = term1 + term2 - factors_log_normaliser

                # log f( sum ) ~ N x 1 x T x 1
                log_sum_exp_n_t = torch.logsumexp(log_prob_n_m_t_r, dim=(1, 3), keepdim=True)

                # log f(E(T(Z))) ~ N x T
                log_denominator_suff_stat = (log_sum_exp_n_t
                                             - log(torch.tensor([num_observations * len_observations],
                                                                dtype=self.dtype,
                                                                device=self.device))) \
                    .squeeze(1).squeeze(-1)

                # Weights ~ N x T x (N' T') should normalize across last dimension
                pi_n_t_mr = torch.exp(log_prob_n_m_t_r - log_sum_exp_n_t) \
                    .permute((0, 2, 1, 3)) \
                    .reshape(num_observations, len_observations, num_observations * len_observations)
                if not ((pi_n_t_mr.sum(dim=-1) - 1).abs().max() < 5e-3):
                    raise ValueError('Incorect Shape normalization for Pi')

                # ~ 1 x 1 x K x N' x T'
                factors_natural1_bis = factors_natural1.permute((0, 2, 4, 1, 3)) \
                    .reshape((1, 1, dim_latent, num_observations * len_observations))
                factors_natural2_bis = factors_natural2.permute((0, 2, 4, 5, 1, 3)) \
                    .reshape((1, 1, dim_latent, dim_latent, num_observations * len_observations))

                # N x T x K x (N' x T')
                XptwiseD1 = factors_natural1_bis * pi_n_t_mr.unsqueeze(2)
                # N x T x K x 1
                XmatmulD1 = matmul(factors_natural1_bis.unsqueeze(-2), pi_n_t_mr.unsqueeze(2).unsqueeze(-1)) \
                    .squeeze(-1)

                # N x T x (K**2) x (N' x T')
                XptwiseD2 = (factors_natural2_bis * pi_n_t_mr.unsqueeze(2).unsqueeze(3)) \
                    .reshape(
                    (num_observations, len_observations, dim_latent * dim_latent, num_observations * len_observations))
                # N x T x (K**2) x 1
                XmatmulD2 = matmul(factors_natural2_bis.unsqueeze(-2),
                                   pi_n_t_mr.unsqueeze(2).unsqueeze(3).unsqueeze(-1)).squeeze(-1) \
                    .reshape((num_observations, len_observations, dim_latent * dim_latent, 1))

                # N x T x K x K
                XdiagDX1 = matmul(XptwiseD1, factors_natural1_bis.transpose(dim0=-1, dim1=-2))
                XDDTX1 = matmul(XmatmulD1, XmatmulD1.transpose(dim0=-1, dim1=-2))
                XPIX1 = XdiagDX1 - XDDTX1

                # N x T x (K**2) x (K**2)
                XdiagDX2 = matmul(XptwiseD2,
                                  factors_natural2_bis
                                  .reshape((1, 1, dim_latent * dim_latent, num_observations * len_observations))
                                  .transpose(dim0=-1, dim1=-2))
                XDDTX2 = matmul(XmatmulD2, XmatmulD2.transpose(dim0=-1, dim1=-2))
                XPIX2 = XdiagDX2 - XDDTX2

                trace1 = (suff_stat_q_variance1 * XPIX1).sum(dim=(-1, -2))
                trace2 = (suff_stat_q_variance2 * XPIX2).sum(dim=(-1, -2))

                log_denominator += log_denominator_suff_stat + 0.5 * (trace1 + trace2)

        return - KLqf - Hq - log_denominator

    def get_free_energy_accurate(self, num_samples=100):
        """ Get a final estimate of the free energy using a ~ high number of samples from the variational marginals"""

        # Constant Term of the free energy
        num_factors = self.num_factors
        num_observation = self.num_observation
        len_observation = self.len_observation
        constant = - len_observation * num_observation * num_factors * log(torch.tensor([num_observation],
                                                                                        dtype=self.dtype,
                                                                                        device=self.device))

        # KL[variational, prior] for Inducing points
        KLqUpU = flexible_kl(self.inducing_points, self.prior, repeat1=[0, 1], repeat2=[1])

        # Re-Define variational marginals to draw sample
        variational_marginals = self.variational_marginals
        self.variational_marginals = FlexibleMultivariateNormal(
            variational_marginals.natural1, variational_marginals.natural2,
            init_natural=True, init_chol=False, use_sample=True, use_suff_stat_mean=True, use_suff_stat_variance=False)

        # Estimate cross entropies that needs samples
        free_energy_theta = self.get_free_energy_theta_samples(num_samples=num_samples)

        # Full free energy
        free_energy = constant - KLqUpU.sum() + free_energy_theta.sum()

        return free_energy

    def init_gp_prior_param(self):
        """ Initialise parameters of k=1..K independent kernels """

        # Number of GP prior
        dim_latent = self.dim_latent

        if self.fit_params['gp_kernel'] == 'RBF':
            # Exponentiated quadratic kernel
            sigma1 = torch.tensor(1 + torch.rand(dim_latent),
                                  dtype=self.dtype, device=self.device, requires_grad=True)
            scale1 = torch.tensor(1 + torch.rand(dim_latent),
                                  dtype=self.dtype, device=self.device, requires_grad=True)
            self.gp_prior_param = (sigma1, scale1)

        elif self.fit_params['gp_kernel'] == 'RBFn':
            # Noisy Exponentiated quadratic kernel
            sigma1 = torch.tensor(1 + torch.rand(dim_latent),
                                  dtype=self.dtype, device=self.device, requires_grad=True)
            scale1 = torch.tensor(1 + torch.rand(dim_latent),
                                  dtype=self.dtype, device=self.device, requires_grad=True)
            scale0 = torch.tensor(1 + torch.rand(dim_latent),
                                  dtype=self.dtype, device=self.device, requires_grad=True)
            self.gp_prior_param = (sigma1, scale1, scale0)

        elif self.fit_params['gp_kernel'] == 'RQ':
            # Rotational Quadratic Kernel
            sigma = torch.tensor(1 + 0 * torch.rand(dim_latent),
                                 dtype=self.dtype, device=self.device, requires_grad=True)
            scale = torch.tensor(1 + 0 * torch.rand(dim_latent),
                                 dtype=self.dtype, device=self.device, requires_grad=True)
            alpha = torch.tensor(1 + 0 * torch.rand(dim_latent),
                                 dtype=self.dtype, device=self.device, requires_grad=True)
            self.gp_prior_param = (sigma, scale, alpha)

        elif 'Matern' in self.fit_params['gp_kernel']:
            # Matern12 / Matern32 / Matern52 Kernel
            sigma = torch.tensor(1 + torch.rand(dim_latent),
                                 dtype=self.dtype, device=self.device, requires_grad=True)
            scale = torch.tensor(1 + torch.rand(dim_latent),
                                 dtype=self.dtype, device=self.device, requires_grad=True)
            self.gp_prior_param = (sigma, scale)

        elif self.fit_params['gp_kernel'] == 'Periodic':
            # Periodic Kernel
            sigma = torch.tensor(1 + 0 * torch.rand(dim_latent),
                                 dtype=self.dtype, device=self.device, requires_grad=True)

            scale = torch.tensor(0.1 + 0 * torch.rand(dim_latent),
                                 dtype=self.dtype, device=self.device, requires_grad=True)

            period = torch.tensor(10 + 0 * torch.rand(dim_latent),
                                  dtype=self.dtype, device=self.device, requires_grad=True)

            self.gp_prior_param = (sigma, scale, period)

        else:
            raise NotImplementedError()

    def init_inducing_points(self):
        """ Initialise the inducing points variational distribution """

        dim_latent = self.dim_latent
        num_observation = self.num_observation
        num_inducing_point = self.num_inducing_point

        inducing_means = torch.randn(num_observation, dim_latent, num_inducing_point,
                                     dtype=self.dtype, device=self.device, requires_grad=True)

        inducing_var_chol_vec_tmp = 2 + np.random.rand(num_observation, dim_latent,
                                                       int(num_inducing_point * (num_inducing_point + 1) / 2))
        inducing_var_chol_vec = torch.tensor(inducing_var_chol_vec_tmp, dtype=self.dtype, device=self.device,
                                             requires_grad=True)

        self.inducing_points_param = (inducing_means, inducing_var_chol_vec)

    def init_factors_tilde(self):
        """ Initialize the auxiliary factors such that the ratio with the factor distributions remains valid """

        if self.fit_params['constraint_factors'] == 'diag':
            # Case where All factors have diagonal covariance matrices

            factors_tilde_params = ()
            for ii in range(self.num_factors):
                factor_shape = self.factors[ii].natural1.shape
                lower_bound = self.factors[ii].natural2.diagonal(dim1=-2, dim2=-1).max(dim=0, keepdim=True)[0]

                natural1_ii = torch.rand(factor_shape, dtype=self.dtype, device=self.device, requires_grad=True)
                natural2_diag_ii_tmp = lower_bound + torch.rand(factor_shape, dtype=self.dtype, device=self.device,
                                                                requires_grad=False)
                natural2_diag_ii = torch.tensor(natural2_diag_ii_tmp,
                                                dtype=self.dtype, device=self.device, requires_grad=True)

                factors_tilde_params += ((natural1_ii, natural2_diag_ii),)

            self.factors_tilde_params = factors_tilde_params

        elif self.fit_params['constraint_factors'] == 'full':

            factors_tilde_params = ()
            for ii in range(self.num_factors):
                factor_shape = self.factors[ii].natural1.shape

                factors_natural2 = self.factors[ii].natural2

                # Orthogonal eigen decompositions
                (eigenvalues, eigenvectors) = torch.linalg.eigh(factors_natural2)
                lower_bound = eigenvalues.max(dim=0, keepdim=True)[0]

                natural1_ii = torch.rand(factor_shape, dtype=self.dtype, device=self.device, requires_grad=True)
                natural2_diag_ii_tmp = lower_bound + torch.rand(factor_shape, dtype=self.dtype, device=self.device,
                                                                requires_grad=False)
                natural2_ii = torch.tensor(diagonalize(natural2_diag_ii_tmp),
                                                dtype=self.dtype, device=self.device, requires_grad=True)

                factors_tilde_params += ((natural1_ii, natural2_ii),)

            self.factors_tilde_params = factors_tilde_params

        else:
            raise NotImplementedError()

    def update_variational_marginals(self):
        """ Update Latent Variational Distribution from Inducing Point Distribution """

        # M Inducing (tau) and T observation (t) locations
        inducing_locations = self.inducing_locations
        observation_locations = self.observation_locations

        # GP Prior Parameters
        gp_prior_param = self.gp_prior_param

        # GP prior applied at inducing location
        gp_prior = self.prior

        # inv Cov_k(tau, tau) ~ K x M x M The inverse covariance is - 2 x natural2
        K_tau_tau_inv = - 2 * gp_prior.natural2

        # Cov_k(tau, tau) ~ K x M x M
        K_tau_tau = self.gp_kernel(inducing_locations, inducing_locations, gp_prior_param)

        # Cov_k(t, tau) ~ K x T x M
        K_t_tau = self.gp_kernel(observation_locations, inducing_locations, gp_prior_param)

        # Cov_k(t,t) ~ K x T (we only keep the diagonal elements)
        K_t_t = self.gp_kernel(observation_locations, observation_locations, gp_prior_param) \
            .diagonal(dim1=-1, dim2=-2)

        # Cov_k(t, tau) inv( Cov_k(tau, tau) ) unsqueezed to ~ 1 x K x T x 1 x M
        K_t_tau_K_tau_tau_inv = matmul(K_t_tau, K_tau_tau_inv).unsqueeze(0).unsqueeze(-2)

        # Mean of the inducing Points ~ N x K x M
        inducing_mean = self.inducing_points.suff_stat_mean[0]

        # Covariance of the inducing points ~ N x K x M x M
        inducing_covariance = self.inducing_points.suff_stat_mean[1] \
                              - matmul(inducing_mean.unsqueeze(-1), inducing_mean.unsqueeze(-2))

        # inducing_covariance - Cov_k(tau, tau) unsqueezed to ~ N x K x 1 x M x M
        delta_K = (inducing_covariance - K_tau_tau.unsqueeze(0)).unsqueeze(-3)

        # Variational Marginals Mean Reshaped and Permuted to ~ N x T x K
        marginal_mean = matmul(
            K_t_tau_K_tau_tau_inv,
            inducing_mean.unsqueeze(-2).unsqueeze(-1)
        ).squeeze(-1).squeeze(-1).permute(0, 2, 1)

        # Variational Marginals Covariance Reshaped and Permuted to ~ N x T x K (note that dimensions are independent)
        marginal_covariance_diag = (
                K_t_t.unsqueeze(0)
                + matmul(matmul(K_t_tau_K_tau_tau_inv, delta_K), K_t_tau_K_tau_tau_inv.transpose(-1, -2))
                .squeeze(-1).squeeze(-1)
        ).permute(0, 2, 1)

        # Make sure entries stay non zero and positive
        # marginal_covariance_diag[(marginal_covariance_diag < 1e-8)] = 1e-8
        marginal_covariance_diag[(marginal_covariance_diag < 1e-6)] = 1e-6

        # Square Root and Diagonalize the marginal Covariance ~ N x T x K x K (Alternatively, use 1D MVN)
        marginal_covariance_chol \
            = torch.zeros(self.num_observation, self.len_observation, self.dim_latent, self.dim_latent,
                          dtype=self.dtype, device=self.device)
        marginal_covariance_chol[..., range(self.dim_latent), range(self.dim_latent)] \
            = torch.sqrt(marginal_covariance_diag)

        # Marginals distributions
        variational_marginals = FlexibleMultivariateNormal(marginal_mean, marginal_covariance_chol,
                                                           init_natural=False, init_chol=True,
                                                           use_sample=self.marginals_use_samples,
                                                           use_suff_stat_mean=True,
                                                           use_suff_stat_variance=self.marginals_use_suff_stat_variance)
        self.variational_marginals = variational_marginals

    def update_prior(self):
        """ Get GP-Prior at the inducing locations """

        # Parameters and inducing locations
        gp_prior_param = self.gp_prior_param
        inducing_locations = self.inducing_locations

        # Mean and Covariance functions
        gp_prior_mean = torch.zeros(self.dim_latent, self.num_inducing_point, dtype=self.dtype, device=self.device)
        gp_prior_covariance = self.gp_kernel(inducing_locations, inducing_locations, gp_prior_param)
        gp_prior = FlexibleMultivariateNormal(gp_prior_mean, gp_prior_covariance,
                                              init_natural=False, init_chol=False)

        self.prior = gp_prior

    def update_factors(self, observations):
        """  Build factor distributions from recognition function output """

        # Latent dimension
        dim_latent = self.dim_latent
        len_observation = self.len_observation
        num_observation = self.num_observation

        # Build factor distributions
        self.factors = ()
        for cur_factor in range(self.num_factors):
            cur_recognition = self.recognition_function[cur_factor]

            # Reshape For conv2d
            unfolded_shape = (num_observation * len_observation, *list(observations[cur_factor].shape[2:]))
            refolded_shape = (num_observation, len_observation, int(dim_latent + dim_latent * (dim_latent + 1) / 2))
            cur_observation = observations[cur_factor].view(unfolded_shape).unsqueeze(1)

            # Recognition Function (+reshaping)
            cur_factors_param = cur_recognition(cur_observation).view(refolded_shape)

            # 1st Natural Parameter
            natural1 = cur_factors_param[..., :dim_latent]

            # Cholesky Decomposition of the (-) second natural parameter
            natural2_chol = vector_to_triul(cur_factors_param[..., dim_latent:])

            # Build factor distributions
            self.factors += (FlexibleMultivariateNormal(natural1, natural2_chol,
                                                        init_natural=True, init_chol=True, use_sample=False,
                                                        use_suff_stat_mean=self.factors_use_suff_stat_mean,
                                                        use_suff_stat_variance=False),)

    def update_inducing_points(self, init=False):
        """ Wrapper to update variational distribution over inducing points """

        if self.fit_params['inference_mode'] == 'MomentMatching' and not init:
            self.update_inducing_points_moment_match()
        else:
            self.update_inducing_points_gradient()

    def update_inducing_points_gradient(self):
        """ Build Inducing points Variational Distributions from natural parameters """

        inducing_means, inducing_var_chol_vec = self.inducing_points_param

        # Positivity Constraint on the diagonal
        num_inducing_point = self.num_inducing_point
        idx_tmp = np.arange(num_inducing_point)
        idx_diag = (num_inducing_point * idx_tmp - idx_tmp * (idx_tmp - 1) / 2).astype(int)
        idx_core = np.setdiff1d(np.arange(inducing_var_chol_vec.shape[-1]), idx_diag)

        inducing_var_chol_vec_cst = torch.zeros(inducing_var_chol_vec.shape, dtype=self.dtype, device=self.device)
        inducing_var_chol_vec_cst[..., idx_diag] = inducing_var_chol_vec[..., idx_diag] ** 2
        inducing_var_chol_vec_cst[..., idx_core] = inducing_var_chol_vec[..., idx_core]

        inducing_var_chol_mat = vector_to_triul(inducing_var_chol_vec_cst)

        inducing_points = FlexibleMultivariateNormal(inducing_means, inducing_var_chol_mat,
                                                     init_natural=False, init_chol=True, use_sample=False,
                                                     use_suff_stat_mean=True, use_suff_stat_variance=False)

        self.inducing_points = inducing_points

    def update_inducing_points_moment_match(self):
        """ Build Inducing points Variational Distributions from Moment Matching """

        dim_latent = self.dim_latent

        # Pevious inducing means ~ N x K x M
        mean_old = self.inducing_points.suff_stat_mean[0]

        # M Inducing (tau) and T observation (t) locations
        inducing_locations = self.inducing_locations
        observation_locations = self.observation_locations

        # GP Prior Parameters
        gp_prior_param = self.gp_prior_param

        # GP prior applied at inducing location
        gp_prior = self.prior

        # inv Cov_k(tau, tau) ~ K x M x M The inverse covariance is - 2 x natural2
        K_tau_tau_inv = - 2 * gp_prior.natural2

        # Cov_k(t, tau) ~ K x T x M
        K_t_tau = self.gp_kernel(observation_locations, inducing_locations, gp_prior_param)

        # Cov_k(t, tau) inv( Cov_k(tau, tau) ) unsqueezed to ~ K x T x M
        K_t_tau_K_tau_tau_inv = matmul(K_t_tau, K_tau_tau_inv)

        # Update the moment matched ratio
        self.moment_match_denominator()

        # Sum the natural parameters of the ratio fi(z|x) / fi(z) over i ~ N x T x K (x K)
        ratio_natural1 = 0
        ratio_natural2 = 0
        num_factors = self.num_factors
        for ii in range(num_factors):
            ratio_natural1 += self.factors_ratio_moment_matched[ii].natural1
            ratio_natural2 += self.factors_ratio_moment_matched[ii].natural2

        # Permute to shape N x K (x K) x T
        ratio_natural1 = ratio_natural1.permute(0, 2, 1)
        ratio_natural2 = ratio_natural2.permute(0, 2, 3, 1)

        # Diag(eta_kl) Cov_l(t, tau) Cov_l(tau, tau)^(-1) ~ N x K x K x T x M
        DKK = ratio_natural2.unsqueeze(-1) * K_t_tau_K_tau_tau_inv.unsqueeze(0).unsqueeze(0)

        # Cov_k(tau, tau)^(-1) Cov_k(tau, t) Diag(eta_kl) Cov_l(t, tau) Cov_l(tau, tau)^(-1) ~ N x K x K x M x M
        KKDKK_kl = matmul(K_t_tau_K_tau_tau_inv.transpose(dim1=-1, dim0=-2).unsqueeze(0).unsqueeze(2),
                          DKK)

        # Cov_k(tau, tau)^(-1) Cov_k(tau, t) Diag(eta_kk) Cov_k(t, tau) Cov_k(tau, tau)^(-1) ~ N x K x M x M
        KKDKK_kk = KKDKK_kl[:, range(dim_latent), range(dim_latent)]

        # Inducing Point Covariances ~ N x K x M x M
        inducing_var = torch.linalg.inv(K_tau_tau_inv - 2 * KKDKK_kk)

        # Temporary Mean ~ N x K x M x 1
        mean_tmp1 = matmul(K_t_tau_K_tau_tau_inv.transpose(dim1=-1, dim0=-2).unsqueeze(0),
                           ratio_natural1.unsqueeze(-1))

        mean_tmp2 = matmul(KKDKK_kl, mean_old.unsqueeze(-1).unsqueeze(1)).sum(dim=2) \
                    - matmul(KKDKK_kk, mean_old.unsqueeze(-1))

        # Inducing Means ~ N x K x M
        inducing_means = matmul(inducing_var, mean_tmp1 + mean_tmp2).squeeze(-1)

        # Build Inducing Point variational distributions
        inducing_points = FlexibleMultivariateNormal(inducing_means, inducing_var,
                                                     init_natural=False, init_chol=False, use_sample=False,
                                                     use_suff_stat_mean=True, use_suff_stat_variance=False)

        self.inducing_points = inducing_points

    def update_factors_tilde(self):
        """ Update the auxiliary factors """

        if self.fit_params['constraint_factors'] == 'diag':
            # Case where All factors have diagonal covariance matrices

            factors_tilde = ()

            for ii in range(self.num_factors):
                factors_tilde_params = self.factors_tilde_params[ii]

                natural1 = factors_tilde_params[0]
                natural2 = diagonalize(factors_tilde_params[1])
                factors_tilde += ((natural1, natural2),)

            self.factors_tilde = factors_tilde

        elif self.fit_params['constraint_factors'] == 'full':
            self.factors_tilde = self.factors_tilde_params
        else:
            raise NotImplementedError()

    def step_prior(self, observations, optimizer_prior, loss_tot):
        """ Take ite_prior prior optimization steps, freeze other parameters """

        for ii in range(self.fit_params['ite_prior']):
            optimizer_prior.zero_grad()
            self.update_all(observations,
                            detach_prior=False, detach_inducing_points=True,
                            detach_factors_tilde=True, detach_recognition=True)
            loss_cur = self.get_loss()
            loss_cur.backward()
            optimizer_prior.step()
            loss_tot = np.append(loss_tot, loss_cur.cpu().detach().numpy()[0])

        return loss_tot

    def step_factors(self, observations, optimizer_factors, loss_tot):
        """ Take ite_factors factor optimization steps, freeze other parameters """

        for ii in range(self.fit_params['ite_factors']):
            optimizer_factors.zero_grad()
            self.update_all(observations,
                            detach_prior=True, detach_inducing_points=True,
                            detach_factors_tilde=True, detach_recognition=False)
            loss_cur = self.get_loss(use_minibatch=True)
            loss_cur.backward()

            optimizer_factors.step()
            loss_tot = np.append(loss_tot, loss_cur.cpu().detach().numpy()[0])

        return loss_tot

    def step_inducing_points(self, observations, optimizer_inducing_points, loss_tot):
        """  Take ite_inducing_points inducing points optimization steps """

        if not (self.fit_params['inference_mode'] == 'MomentMatching'):
            for ii in range(self.fit_params['ite_inducing_points']):
                optimizer_inducing_points.zero_grad()
                self.update_all(observations,
                                detach_prior=True, detach_inducing_points=False,
                                detach_factors_tilde=True, detach_recognition=True)
                loss_cur = self.get_loss()
                loss_cur.backward()
                optimizer_inducing_points.step()
                loss_tot = np.append(loss_tot, loss_cur.cpu().detach().numpy()[0])

        else:
            # In the moment matching case, inducing point distribution is closed form
            optimizer_inducing_points.zero_grad()
            loss_cur = self.get_loss()
            loss_tot = np.append(loss_tot, loss_cur.cpu().detach().numpy()[0])

        return loss_tot

    def step_factors_tilde(self, observations, optimizer_factors_tilde, loss_tot):
        """ Take ite_factors_tilde auxiliary factors optimization steps """

        for ii in range(self.fit_params['ite_factors_tilde']):
            optimizer_factors_tilde.zero_grad()
            self.update_all(observations,
                            detach_prior=True, detach_inducing_points=True,
                            detach_factors_tilde=False, detach_recognition=True)
            loss_cur = self.get_loss()
            loss_cur.backward()
            optimizer_factors_tilde.step()
            loss_tot = np.append(loss_tot, loss_cur.cpu().detach().numpy()[0])

        return loss_tot

    def update_all(self, observations, detach_prior=False, detach_inducing_points=False,
                   detach_factors_tilde=False, detach_recognition=False):
        """ Wrapper to handle RP-GPFA fit in an Expectation Maximization way"""

        with torch.set_grad_enabled(not detach_recognition):
            self.update_factors(observations)

        if self.fit_params['inference_mode'] == 'VariationalBound':
            with torch.set_grad_enabled(not detach_factors_tilde):
                self.update_factors_tilde()

        with torch.set_grad_enabled(not detach_prior):
            self.update_prior()

        with torch.set_grad_enabled(not detach_inducing_points):
            self.update_inducing_points()

        self.update_variational_marginals()

    def init_fit_params(self):
        """ Default Fit parameters """

        # Ergodicity Assumptions (for all factor j: p0j(xj) = p0(xj))
        if not ('ergodic' in self.fit_params.keys()):
            self.fit_params['ergodic'] = False

        # Recognition Network(s) structure
        if not ('dim_hidden' in self.fit_params.keys()):
            # Init with similar network for each factors
            self.fit_params['dim_hidden'] = tuple([[50, 50] for _ in range(self.num_factors)])
        else:
            # Check that for each factor hidden layer dimensions are provided
            assert len(self.fit_params['dim_hidden']) == self.num_factors

        if not ('nn_type' in self.fit_params.keys()):
            # Init with similar network for each factors
            self.fit_params['nn_type'] = tuple(['feedforward' for _ in range(self.num_factors)])
        else:
            # Check that for each factor type are provided
            assert len(self.fit_params['nn_type']) == self.num_factors

        # Latent dimensions
        if not ('dim_latent' in self.fit_params.keys()):
            self.fit_params['dim_latent'] = 1

        # Inference Method (Samples - 2ndOrder - VariationalBound) and constraints
        if not ('inference_mode' in self.fit_params.keys()):
            self.fit_params['inference_mode'] = '2ndOrder'
        if not ('constraint_factors' in self.fit_params.keys()):
            if self.fit_params['inference_mode'] == 'VariationalBound':
                self.fit_params['constraint_factors'] = 'diag'
            else:
                self.fit_params['constraint_factors'] = 'full'

        # GP kernel
        if not ('gp_kernel' in self.fit_params.keys()):
            self.fit_params['gp_kernel'] = 'RBF'

        # Iterations
        if not ('ite_out' in self.fit_params.keys()):
            self.fit_params['ite_out'] = 500
        if not ('ite_prior' in self.fit_params.keys()):
            self.fit_params['ite_prior'] = 50
        if not ('ite_inducing_points' in self.fit_params.keys()):
            self.fit_params['ite_inducing_points'] = 50
        if not ('ite_factors' in self.fit_params.keys()):
            self.fit_params['ite_factors'] = 50
        if not ('ite_factors_tilde' in self.fit_params.keys()):
            self.fit_params['ite_factors_tilde'] = 50

        # Optimizers
        if not ('optimizer_prior' in self.fit_params.keys()):
            self.fit_params['optimizer_prior'] = \
                {'name': 'Adam', 'param': {'lr': 1e-3}}
        if not ('optimizer_inducing_points' in self.fit_params.keys()):
            self.fit_params['optimizer_inducing_points'] = \
                {'name': 'Adam', 'param': {'lr': 1e-3}}
        if not ('optimizer_factors' in self.fit_params.keys()):
            self.fit_params['optimizer_factors'] = \
                {'name': 'Adam', 'param': {'lr': 1e-3}}
        if not ('optimizer_factors_tilde' in self.fit_params.keys()):
            self.fit_params['optimizer_factors_tilde'] = \
                {'name': 'Adam', 'param': {'lr': 1e-3}}

        # Sampling Specific Param
        if not ('num_samples' in self.fit_params.keys()):
            self.fit_params['num_samples'] = 30

    def moment_match_denominator(self):

        num_factors = self.num_factors
        num_observation = self.num_observation

        # Normalizing constants
        log_beta = ()

        # Natural Parameters of the moment matched ratios ~ N x T x K (x K)
        factors_ratio_moment_matched = ()

        for ii in range(num_factors):
            # Recognition Factor i
            factors = self.factors[ii]

            # Mean off the sufficient Statistics ~ N x T x K (x K)
            factors_suff_stat1 = factors.suff_stat_mean[0]
            factors_suff_stat2 = factors.suff_stat_mean[1]

            # Moment Matched Denominator ~ 1 x T x K (x K)
            mm_mean = factors_suff_stat1.sum(dim=0) / num_observation
            mm_variance = factors_suff_stat2.sum(dim=0) / num_observation \
                          - matmul(mm_mean.unsqueeze(-1), mm_mean.unsqueeze(-2))
            mm_denominator = FlexibleMultivariateNormal(mm_mean, mm_variance, init_natural=False, init_chol=False)

            # Moment Matched Ratio Natural Parameter ~ N x T x K (x K)
            ratio_i_natural1 = factors.natural1 - mm_denominator.natural1.unsqueeze(0)
            ratio_i_natural2 = factors.natural2 - mm_denominator.natural2.unsqueeze(0)

            # Ensure the ratio to be a valid distribution
            ratio_i_natural2 = threshold_eigh(ratio_i_natural2, bound=1e-3)

            # Moment Matched Ratio
            mm_ratio_i = FlexibleMultivariateNormal(ratio_i_natural1, ratio_i_natural2, init_natural=True,
                                                    init_chol=False)

            # Store Distribution
            factors_ratio_moment_matched += (mm_ratio_i,)

            # Grasp Normaliser
            factors_log_normalizer = factors.log_normalizer
            mm_ratio_log_normalizer = mm_ratio_i.log_normalizer
            mm_denominator_log_normalizer = mm_denominator.log_normalizer.unsqueeze(0)

            # Normalizing constants ~ N x T
            log_beta_i = mm_ratio_log_normalizer - (factors_log_normalizer - mm_denominator_log_normalizer)
            log_beta += (log_beta_i,)

        self.factors_ratio_moment_matched = factors_ratio_moment_matched
        self.factors_ratio_log_beta = log_beta

    def get_free_energy_theta_moment_match(self):

        # Conditionally independent factors
        num_factors = self.num_factors

        # Grasp variational marginals over latent
        variational_marginals = self.variational_marginals
        Hq = num_factors * variational_marginals.entropy

        # Init log(beta) and KL(q||ratio) ~ N x T
        KLqratio = 0
        log_beta = 0

        for ii in range(num_factors):
            # Normalizing constant for the rations
            log_beta_i = self.factors_ratio_log_beta[ii]

            # KL(q||ratio_i)
            ratio_i = self.factors_ratio_moment_matched[ii]
            KLqratio_i = flexible_kl(variational_marginals, ratio_i)

            # Sum over factors
            log_beta += log_beta_i
            KLqratio += KLqratio_i

        return -Hq + log_beta - KLqratio


def save_gprpm(model, observations, observation_locations, filename, true_latent=None, convert_to_cpu=False):
    """ Helper to save a RP-GPFA model """

    with open(filename, 'wb') as outp:

        if convert_to_cpu:
            # Move all objects from GPU to CPU

            observations_cpu = ()
            for obsi in observations:
                observations_cpu += (obsi.to("cpu"),)

            recognition_function_cpu = ()
            for reci in model.recognition_function:
                recognition_function_cpu += (reci.to("cpu"),)

            observation_locations_cpu = observation_locations.to("cpu")
            inducing_locations_cpu = model.inducing_locations.to("cpu")

            fit_params_cpu = model.fit_params
            inducing_points_param_cpu = tuple([i.to("cpu") for i in model.inducing_points_param])
            gp_prior_param_cpu = tuple([i.to("cpu") for i in model.gp_prior_param])

            if not model.factors_tilde_params is None:
                factors_tilde_params_cpu = tuple([(i[0].to("cpu"), i[1].to("cpu")) for i in model.factors_tilde_params])
            else:
                factors_tilde_params_cpu = None

            if not true_latent is None:
                true_latent_cpu = true_latent.to("cpu")
            else:
                true_latent_cpu = None

        else:
            # Stay on CPU
            observations_cpu = observations
            observation_locations_cpu = observation_locations
            inducing_locations_cpu = model.inducing_locations
            fit_params_cpu = model.fit_params
            inducing_points_param_cpu = model.inducing_points_param
            gp_prior_param_cpu = model.gp_prior_param
            recognition_function_cpu = model.recognition_function
            factors_tilde_params_cpu = model.factors_tilde_params
            true_latent_cpu = true_latent

        # Save as Dict
        model_save = {'observations': observations_cpu,
                      'observation_locations': observation_locations_cpu,
                      'inducing_locations': inducing_locations_cpu,
                      'fit_params': fit_params_cpu,
                      'inducing_points_param': inducing_points_param_cpu,
                      'gp_prior_param': gp_prior_param_cpu,
                      'recognition_function': recognition_function_cpu,
                      'factors_tilde_params': factors_tilde_params_cpu,
                      'loss_tot': model.loss_tot,
                      'true_latent': true_latent_cpu}

        pickle.dump(model_save, outp, pickle.HIGHEST_PROTOCOL)


def load_gprpm(model_name):
    """ Helper to Load a RP-GPFA model """
    with open(model_name, 'rb') as outp:
        model_tmp = pickle.load(outp)

        # Temporary wrapper for all models
        if 'loss_tot' in model_tmp.keys():
            loss_tot = model_tmp['loss_tot']
        else:
            loss_tot = 0

        if 'true_latent' in model_tmp.keys():
            true_latent = model_tmp['true_latent']
        else:
            true_latent = None

        observations = model_tmp['observations']
        observation_locations = model_tmp['observation_locations']

        # Reconstruct Model
        model_loaded = UnstructuredRecognition(observations, observation_locations,
                                               inducing_locations=model_tmp['inducing_locations'],
                                               fit_params=model_tmp['fit_params'],
                                               inducing_points_param=model_tmp['inducing_points_param'],
                                               gp_prior_param=model_tmp['gp_prior_param'],
                                               recognition_function=model_tmp['recognition_function'],
                                               factors_tilde_params=model_tmp['factors_tilde_params'])
    return model_loaded, observations, observation_locations, loss_tot, true_latent
