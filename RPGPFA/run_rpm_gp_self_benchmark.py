#%% Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import generate_2D_latent, generate_skewed_pixel_from_latent
from unstructured_recognition_gpfa import UnstructuredRecognition
from datetime import datetime
from unstructured_recognition_gpfa import save_gprpm, load_gprpm

# Generate Data
data_type = torch.float32
torch.set_default_dtype(data_type)

# GPUs ?
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('GP-RPM on GPU')
else:
    print('GP-RPM on CPU')


# Dimension of the observations
num_observation = 50
dim_observation = 10
len_observation = 50
num_inducing = 20

GPkernel = 'RBF'

ite_out = 800
xp_name = 'standard'
scale_th = 0.15
shape_max_0 = 1000
sigma2 = 0.01

# Sampling Frequency
F = 10

# Length of Each sample [sec]
T = int(len_observation / F)

# Oscillation Speed
omega = 0.5


# Colors pixels
color_position = torch.cat((
    torch.linspace(0, 0, dim_observation).unsqueeze(1),
    torch.linspace(0, 1, dim_observation).unsqueeze(1),
    torch.linspace(0, 0, dim_observation).unsqueeze(1)),
    dim=1)


# Random initializations
theta = 2*np.pi*np.random.rand(num_observation)
z0 = torch.tensor(np.array([np.cos(theta), np.sin(theta)]).T)
zt, _ = generate_2D_latent(T, F, omega, z0)


check_distrib = False
if check_distrib:
    N = 100000
    latent_test = torch.empty(N, 3, 1)
    latent_test[:, 0, 0] = 1
    latent_test[:, 1, 0] = 0
    latent_test[:, 2, 0] = -1
    observation_samples = generate_skewed_pixel_from_latent(latent_test, 10,
                                                            scale_th=scale_th, sigma2=sigma2, shape_max_0=shape_max_0)

    plt.figure(figsize=(4, 4))
    s1 = observation_samples[:, 0, 9]
    s2 = observation_samples[:, 0, 0]
    plt.hist(s1, bins=500, density=True, alpha=0.75, label='Ball Close')
    plt.hist(s2, bins=500, density=True, alpha=0.75, label='Ball Far')
    plt.xlabel('Pixel Intensity')
    plt.xlabel('Intensity Distribution')
    plt.legend()
    print('Mean1: %.2e' % s1.mean() + '| Var1: %.2e' % s1.var())
    print('Mean2: %.2e' % s2.mean() + '| Var2: %.2e' % s2.var())

# True Latent
true_latent_ext = zt[:, 1:, 0] .unsqueeze(-1)

# Sample Observations
observation_samples = generate_skewed_pixel_from_latent(true_latent_ext, dim_observation,
                                                    scale_th=scale_th, sigma2=sigma2, shape_max_0=shape_max_0)

# Convert Observations
observations = torch.tensor(observation_samples, dtype=data_type, device=device)

# Set up observation and inducing points locations
observation_locations = torch.arange(0, len_observation, 1, dtype=data_type, device=device).unsqueeze(-1)
inducing_locations = torch.arange(0, len_observation, len_observation / num_inducing, dtype=data_type, device=device)\
                          .unsqueeze(-1)

# Plot Observation
do_plot = not(torch.cuda.is_available())
plot_index = [0, 1, 2, 3]
egpix = 0
ccmap = 'gray'
pixel_plot = [0, int(dim_observation / 4)  , 4 * int(dim_observation / 4) -1  ]
plot_observations = not(torch.cuda.is_available())

if do_plot:
    plt.figure(figsize=(3 * len(plot_index), 2 * 2))
    for egobs_id in range(len(plot_index)):
        egobs = plot_index[egobs_id]

        plt.subplot(2, len(plot_index), egobs_id + 1)
        plt.imshow(observation_samples[egobs].transpose(-1, -2), aspect='auto', cmap=ccmap,
                   extent=[0, 1, -1, 1], vmin=-1, vmax = 1)
        plt.title('Example ' + str(egobs) + '/' + str(num_observation))
        if egobs_id == 0:
            plt.ylabel('Pixel Id.')

        # Plot All Pixels in 2D
        plt.subplot(2, len(plot_index), egobs_id + 1 + len(plot_index))
        for dd in (np.arange(dim_observation)):
            plt.scatter(np.arange(len_observation) / len_observation, observations[egobs, :, dd], color=color_position[dd].numpy(),
                        label=str(dd))
        if egobs_id == len(plot_index)-1:
            plt.legend(title='Pixel Id.')
        plt.xlabel('Time [a.u]')
        plt.grid()
        plt.autoscale(enable=True, axis='x', tight=True)

        if egobs_id == 0:
            plt.ylabel('Pixel Intensity')

observations = (observations,)


#%% Fit and Save



fit_params_2ndOrder = {'dim_latent': 1,
               'inference_mode': '2ndOrder',
               'constraint_factors': 'full',
               'ite_out': ite_out,
               'ite_prior': 50,
               'ite_inducing_points': 50,
               'ite_factors': 50,
               'optimizer_prior': {'name': 'Adam', 'param': {'lr': 0.5e-3}},
               'optimizer_factors': {'name': 'Adam', 'param': {'lr': 1e-3}},
               'optimizer_inducing_points': {'name': 'Adam', 'param': {'lr': 1e-3}},
               'gp_kernel': GPkernel,
               'dim_hidden': ([50, 50],)
                }

fit_params_VariationalBound = {'dim_latent': 1,
               'inference_mode': 'VariationalBound',
               'constraint_factors': 'diag',
               'ite_out': ite_out,
               'ite_prior': 50,
               'ite_inducing_points': 50,
               'ite_factors': 50,
               'optimizer_prior': {'name': 'Adam', 'param': {'lr': 0.5e-3}},
               'optimizer_factors': {'name': 'Adam', 'param': {'lr': 1e-3}},
               'optimizer_inducing_points': {'name': 'Adam', 'param': {'lr': 1e-3}},
               'gp_kernel': GPkernel,
               'dim_hidden': ([50, 50],)
                }

# Do not delete: works for samples
fit_params_Samples = {'dim_latent': 1,
              'inference_mode': 'Samples',
              'constraint_factors': 'full',
              'ite_out': ite_out,
              'ite_prior': 50,
              'ite_inducing_points': 50,
              'ite_factors': 50,
              'optimizer_prior': {'name': 'Adam', 'param': {'lr': 1e-3}},
              'optimizer_factors': {'name': 'Adam', 'param': {'lr': 1e-3}},
              'optimizer_inducing_points': {'name': 'Adam', 'param': {'lr': 1e-3}},
              'gp_kernel': 'RBF',
              'num_samples': 20}



#%% Fits
trial_max = 10
random_id = str(int(10000*torch.rand(1).numpy()))


# 2nd Order
trial_num = 0
while trial_num<trial_max:
     try:
         model_2ndOrder = UnstructuredRecognition(observations, observation_locations,
                                                  inducing_locations=inducing_locations, fit_params=fit_params_2ndOrder)
         loss_tot = model_2ndOrder.fit(observations)
         break
     except ValueError:
         trial_num += 1
         print("Fit with a new init. " + str(trial_num) + '/' + str(trial_max) )
model_name = '../results_gp_rpm/gp_rpm_benchmark_skew_'+ xp_name + '_' + GPkernel + 'id' + random_id + '_' + '_2ndOrder_'  + datetime.now().strftime("%Y_%M_%d_%Hh%Mm%Ss") + '.pkl'
print("Saving: ", model_name)
save_gprpm(model_2ndOrder, observations, observation_locations, model_name, true_latent=true_latent_ext, convert_to_cpu=torch.cuda.is_available())


# Sampling
trial_num = 0
while trial_num<trial_max:
     try:
         model_Samples = UnstructuredRecognition(observations, observation_locations,
                                                 inducing_locations=inducing_locations, fit_params=fit_params_Samples)
         loss_tot = model_Samples.fit(observations)
         break
     except ValueError:
         trial_num += 1
         print("Fit with a new init. " + str(trial_num) + '/' + str(trial_max) )
model_name = '../results_gp_rpm/gp_rpm_benchmark_skew_'+ xp_name + '_' + GPkernel + 'id' + random_id + '_'  + '_Samples_'  + datetime.now().strftime("%Y_%M_%d_%Hh%Mm%Ss") + '.pkl'
print("Saving: ", model_name)
save_gprpm(model_Samples, observations, observation_locations, model_name, true_latent=true_latent_ext, convert_to_cpu=torch.cuda.is_available())


# DV-Bound
trial_num = 0
while trial_num<trial_max:
     try:
         model_VariationalBound = UnstructuredRecognition(observations, observation_locations,
                                                          inducing_locations=inducing_locations,
                                                          fit_params=fit_params_VariationalBound)
         loss_tot = model_VariationalBound.fit(observations)
         break
     except ValueError:
         trial_num += 1
         print("Fit with a new init. " + str(trial_num) + '/' + str(trial_max) )
model_name = '../results_gp_rpm/gp_rpm_benchmark_skew_'+ xp_name + '_' + GPkernel + 'id' + random_id + '_' + '_VariationalBound_'  +  datetime.now().strftime("%Y_%M_%d_%Hh%Mm%Ss") + '.pkl'
print("Saving: ", model_name)
save_gprpm(model_VariationalBound, observations, observation_locations, model_name, true_latent=true_latent_ext, convert_to_cpu=torch.cuda.is_available())


