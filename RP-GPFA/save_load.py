import torch
import pickle
from recognition_parametrised_gpfa import RPGPFA


def save_gprpm(filename, model, observations=None, true_latent=None):
    """ Helper to save a RP-GPFA model (converts all objects to cpu)"""

    with open(filename, 'wb') as outp:

        # Factors Neural Network
        recognition_function_cpu = ()
        for reci in model.recognition_function:
            recognition_function_cpu += (reci.to("cpu"),)

        # Inducing and observation locations
        inducing_locations_cpu = model.inducing_locations.to("cpu")
        observation_locations_cpu = model.observation_locations.to("cpu")

        # Fit parameters
        fit_params_cpu = model.fit_params

        # Natural Parameters of the inducing points
        inducing_points_param_cpu = tuple([i.to("cpu") for i in model.inducing_points_param])

        # Prior Kernel Parameters
        # kernel_parameters_cpu = tuple([i.to("cpu") for i in model.kernel.parameters()])
        prior_covariance_kernel_cpu = model.prior_covariance_kernel.to("cpu")

        # Prior Mean Parameters
        prior_mean_param_cpu = tuple([i.to("cpu") for i in model.prior_mean_param])

        # Observations
        observations_cpu = None
        if not (observations is None):
            observations_cpu = ()
            for obsi in observations:
                observations_cpu += (obsi.to("cpu"),)

        # True Underlying latent (if known)
        true_latent_cpu = None
        if not (true_latent is None):
            true_latent_cpu = true_latent.to("cpu")

        # Store as Dictionary
        model_save = {'loss_tot': model.loss_tot,
                      'fit_params': fit_params_cpu,
                      'true_latent': true_latent_cpu,
                      'observations': observations_cpu,
                      'prior_mean_param': prior_mean_param_cpu,
                      'prior_covariance_kernel': prior_covariance_kernel_cpu,
                      'recognition_function': recognition_function_cpu,
                      'inducing_points_param': inducing_points_param_cpu,
                      'inducing_locations': inducing_locations_cpu,
                      'observation_locations': observation_locations_cpu}

        # Save
        pickle.dump(model_save, outp, pickle.HIGHEST_PROTOCOL)


def load_gprpm(model_name, observations=None):
    """ Helper to Load a RP-GPFA model """
    with open(model_name, 'rb') as outp:
        model_parameters = pickle.load(outp)

        # Current Loss, fit params, latent and Observations (if provided)
        loss_tot = model_parameters['loss_tot']
        fit_params = model_parameters['fit_params']
        true_latent = model_parameters['true_latent']
        observations = model_parameters['observations'] if observations is None else observations

        # Current Parameter Estimates
        prior_mean_param = model_parameters['prior_mean_param']
        prior_covariance_kernel = model_parameters['prior_covariance_kernel']
        recognition_function = model_parameters['recognition_function']
        inducing_points_param = model_parameters['inducing_points_param']

        # Observations and Inducing locations
        inducing_locations = model_parameters['inducing_locations']
        observation_locations = model_parameters['observation_locations']

        model_loaded = RPGPFA(
            observations, observation_locations, inducing_locations=inducing_locations,
            fit_params=fit_params, loss_tot=loss_tot,
            prior_mean_param=prior_mean_param, prior_covariance_kernel=prior_covariance_kernel,
            inducing_points_param=inducing_points_param, recognition_function=recognition_function)

    return model_loaded, observations, true_latent
