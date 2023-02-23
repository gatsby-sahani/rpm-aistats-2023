#%% Imports
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import imageio
import pickle
from utils import get_speech_samples
from unstructured_recognition_gpfa import UnstructuredRecognition, save_gprpm, load_gprpm

# Reproducibility
# np.random.seed(1)
# torch.manual_seed(1)

ergodic = False
use_sound_speech = False
num_observation = 3 # 20
num_inducing = 50
len_snippet = 100
dim_latent = 2


# GPUs ?
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('GP-RPM on GPU')
else:
    print('GP-RPM on CPU')

# Data type: float64 / float32
data_type = torch.float32
torch.set_default_dtype(data_type)

# Stored Videos
trajectory_folder = '../videos_rpm/'
trajectory_name = 'trajectories_param.pkl'

# Load Trajectories
with open(trajectory_folder + trajectory_name, "rb") as input_file:
        trajectories = pickle.load(input_file)
spatial_trajectories = trajectories['spatial_trajectories']
distance_from_fixed = trajectories['distance_from_fixed']
main_agent = trajectories['main_agent']
video_tensor = trajectories['video_tensor']
main_trajectory = spatial_trajectories[:, main_agent]


num_observations_full, len_observation, _, _ = video_tensor.shape

# Plot videos and sound recordings for one observation (if not on GPU)
plot_observations = not(torch.cuda.is_available())
if plot_observations:

    # Microphone Locations
    mic_position = torch.tensor([[-1, -1], [-1, +1], [+1, -1], [+1, +1]], device=device, dtype=data_type)
    mic_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    # 'Color' of the agents
    c1 = [0.8, 0.8, 0.8]
    c2 = [0.5, 0.5, 0.5]
    c3 = [0.3, 0.3, 0.3]
    color_agents = [c1, c2, c3]

    obs_eg = 6
    tplot = [0, 100, 150]

    pheigh = int(np.floor(np.sqrt(len(tplot)+2)))
    pwidth = int(np.ceil((len(tplot)+2) / pheigh))

    pheigh = 1
    pwidth = int(np.ceil((len(tplot) + 2) / pheigh))

    plt.figure(figsize=(pwidth * 4, pheigh * 4))

    for tt_id in range(len(tplot)):
        name = 'video_' + 'full_plot_n' + str(obs_eg).zfill(3) + '_t' + str(tplot[tt_id]).zfill(4) + '.png'
        im = imageio.imread(trajectory_folder + name)
        print(name)
        plt.subplot(pheigh, pwidth, tt_id + 1)
        plt.imshow(im)
        plt.imshow(video_tensor[obs_eg, tplot[tt_id]], cmap='gray')

        plt.title('t = ' + str(tplot[tt_id]) + '/' + str(len_observation))
        plt.xlim([75, 550])
        plt.ylim([550, 75])
        plt.axis('off')

    plt.subplot(pheigh, pwidth, tt_id + 2)
    for ii in range(mic_position.shape[0]):
        plt.scatter(mic_position[ii, 0], mic_position[ii, 1], s=100, c=mic_colors[ii], marker='s')
    plt.plot(main_trajectory[obs_eg, :, 0], main_trajectory[obs_eg, :, 1], c=color_agents[main_agent])
    plt.scatter(main_trajectory[obs_eg, 0, 0], main_trajectory[obs_eg, 0, 1], s=100, c=color_agents[main_agent],
                label='Start')
    plt.title('Top View')
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    plt.xlabel('z1[t]')
    plt.ylabel('z2[t]')
    plt.legend()

    plt.subplot(pheigh, pwidth, tt_id + 3)
    for ii in range(mic_position.shape[0]):
        plt.plot(np.linspace(0, 1, len_observation), distance_from_fixed[obs_eg, :, ii], c=mic_colors[ii],
                    label='mic.' + str(ii))
    plt.legend(loc=5)
    plt.title('Distance(mic.)')
    plt.xlabel('Time [a.u]')


#%%

audio_path = './../timit/train/'

# return of size  x T x 4 x len_snippet
distance_modulated_audio, time_snippet_sec = get_speech_samples(distance_from_fixed, audio_path,
                                                                downsample=20, len_snippet=len_snippet, normalize=True)
#%% Fit

time_locations = torch.arange(0, len_observation, device=device, dtype=data_type).unsqueeze(-1)
inducing_locations = torch.linspace(0, len_observation,   num_inducing, device=device, dtype=data_type)\
    .unsqueeze(-1)

# Build Observed Factors (Transfer to GPU if necessary)
observations1 = torch.tensor(video_tensor[:num_observation, :, 1:][..., 1:-2], device=device, dtype=data_type)
observations1 = (1 - observations1 / observations1.max())


if use_sound_speech:
    distance_modulated_audio = distance_modulated_audio[:num_observation].reshape(num_observation, len_observation, -1)
    observations2 = torch.tensor(distance_modulated_audio, device=device, dtype=data_type)
else:
    observations2 = torch.tensor(distance_from_fixed[:num_observation], device=device, dtype=data_type)

observations = (observations1, observations2)
observation_locations = time_locations
inducing_locations = torch.linspace(0, len_observation,   num_inducing, device=device, dtype=data_type)\
    .unsqueeze(-1)


fit_params = {'dim_latent': 2,
               'inference_mode': 'VariationalBound',
               'constraint_factors': 'full',
               'ite_out': 800,
               'ite_prior': 50,
               'ite_inducing_points': 50,
               'ite_factors': 50,
               'optimizer_prior': {'name': 'Adam', 'param': {'lr': 0.5e-3}},
               'optimizer_factors': {'name': 'Adam', 'param': {'lr': 1e-3}},
               'optimizer_inducing_points': {'name': 'Adam', 'param': {'lr': 0.5e-3}},
               'gp_kernel': 'RBF',
               'dim_hidden': ([40, 40], [40, 40]), # was all 20
                'nn_type': ('convolutional', 'feedforward'),
                'ergodic': ergodic
                }

model = UnstructuredRecognition(observations, observation_locations,
                                inducing_locations=inducing_locations,
                                fit_params=fit_params)

print('RP-GPFA Video + Sound - Speech=' + str(use_sound_speech))
print('Fit params')
print('________________________________________________________')
for key, value in fit_params.items():
    print(key, ':', value)
print('________________________________________________________')

model.fit(observations)


# Save Model
model_name = '../results_gp_rpm/gpfa_video_speech_rpm_speech' + str(use_sound_speech) + '_' + datetime.now().strftime("%Y_%M_%d_%Hh%Mm%Ss") + '.pkl'
print("Saving: ", model_name)

save_gprpm(model, observations, observation_locations, model_name,
           true_latent=main_trajectory,
           convert_to_cpu=torch.cuda.is_available())

