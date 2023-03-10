#%% Imports
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import imageio
import pickle
from recognition_parametrised_gpfa import RPGPFA
from utils import diagonalize

# Reproducibility
np.random.seed(1)
torch.manual_seed(1)

ergodic = False
use_sound_speech = False
num_observation = 10 #20
num_inducing = 50
len_snippet = 100
dim_latent = 2


# GPUs ?
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
#%% Fit

time_locations = torch.arange(0, len_observation, device=device, dtype=data_type).unsqueeze(-1)
inducing_locations = torch.linspace(0, len_observation,   num_inducing, device=device, dtype=data_type)\
    .unsqueeze(-1)

# Build Observed Factors (Transfer to GPU if necessary)
observations1 = torch.tensor(video_tensor[:num_observation, :, 1:][..., 1:-2], device=device, dtype=data_type)
observations1 = (1 - observations1 / observations1.max())
observations2 = torch.tensor(distance_from_fixed[:num_observation], device=device, dtype=data_type)


#%%

def normalize_observations(obs, num_event_dim=1):

    full_dim = obs.shape
    batch_dim = torch.tensor(obs.shape[:num_event_dim])
    event_dim = torch.tensor(obs.shape[num_event_dim:])

    obs = obs.reshape(batch_dim.prod(), event_dim.prod())
    o_mean, o_std = torch.mean(obs, dim=0, keepdim=True), torch.std(obs, dim=0, keepdim=True)
    obs = (obs - o_mean) / (o_std + 1e-6)
    obs = obs.reshape(full_dim)

    return obs

observations1 = normalize_observations(observations1, num_event_dim=2)
observations2 = normalize_observations(observations2, num_event_dim=2)



observations = (observations1, observations2)
observation_locations = torch.linspace(0, 1, len_observation, device=device, dtype=data_type).unsqueeze(-1)
inducing_locations = torch.linspace(0, 1,   num_inducing, device=device, dtype=data_type).unsqueeze(-1)

fit_params = {'dim_latent': 2,
               'inference_mode': 'VariationalBound',
               'constraint_factors': 'fixed_diag',
               'num_epoch': 20000,
               'optimizer_prior': {'name': 'Adam', 'param': {'lr': 1e-3}},
               'optimizer_factors': {'name': 'Adam', 'param': {'lr': 1e-4}},
               'optimizer_inducing_points': {'name': 'Adam', 'param': {'lr': 1e-3}},
               'gp_kernel': 'RBF',
               'dim_hidden': ([50, 50], [50, 50]), # was all 20
               'nn_type': ('convolutional', 'perceptron'),
               'minibatch_size': len_observation,
               'ergodic': False
                }


model = RPGPFA(observations, observation_locations,
               inducing_locations=inducing_locations,
               fit_params=fit_params)



print('RP-GPFA Video + Sound - Speech=' + str(use_sound_speech))
print('Fit params')
print('________________________________________________________')
for key, value in fit_params.items():
    print(key, ':', value)
print('________________________________________________________')

model.fit(observations)






model_name = 'tmp'
latent_true = main_trajectory[:num_observation]



#%%

from utils_process import plot_summary, plot_factors_prior, plot_loss


plot_loss(model)
plot_factors_prior(model)
plot_summary(model, latent_true=latent_true, plot_observation=[0], plot_factors_id= 'all', plot_regressed='linear',
             plot_true=True)




#%%



from kernels import RBFKernel


plot_summary(model, observations, latent_true, plot_factor_id=-1, plot_regressed=False, plot_n=None)




#%%







#%% Plot Loss
offset = 0
plt.figure()
plt.plot(model.loss_tot[offset:])
plt.xlabel('Iterations')
plt.ylabel('Free Energy')


#%% Plot infered Latent

from torch import matmul
from utils_process import custom_procurustes, plot_ellipses

# Dimensions
dim_latent = model.dim_latent
num_factors = model.num_factors
num_observation = model.num_observation
len_observation = model.len_observation

# Latent Mean and variance
Zt = model.variational_marginals.suff_stat_mean[0].detach().clone()
St = model.variational_marginals.suff_stat_mean[1].detach().clone().diagonal(dim1=-1, dim2=-2) - Zt**2
dim_latent = Zt.shape[-1]

# Reshape and reorder fit and target
latent_true = main_trajectory[:num_observation]
latent_mean_fit = Zt
latent_variance_fit = diagonalize(St)

do_procrustres = True
# Rotate and Rescale
if do_procrustres:

    shape_cur = (num_observation, len_observation, dim_latent)
    shape_tmp = (num_observation * len_observation, dim_latent)

    # Reshape And Diagonalize
    latent_true = main_trajectory[:num_observation].reshape(shape_tmp)
    latent_mean_fit = latent_mean_fit.reshape(shape_tmp)
    latent_variance_fit = latent_variance_fit.reshape((*shape_tmp, dim_latent))

    # Procrustes Transformation
    latent_true, latent_mean_fit, _, _, R2tot = custom_procurustes(latent_true, latent_mean_fit.numpy())
    R2Ttot = torch.tensor(R2tot.T, dtype=model.dtype).unsqueeze(0)
    R2tot = torch.tensor(R2tot, dtype=model.dtype).unsqueeze(0)

    # Back to torch
    latent_true = torch.tensor(latent_true, dtype=model.dtype)
    latent_mean_fit = torch.tensor(latent_mean_fit, dtype=model.dtype)
    latent_variance_fit = matmul(R2Ttot, matmul(latent_variance_fit, R2tot))

    # Reshape
    latent_true = latent_true.reshape(shape_cur)
    latent_mean_fit = latent_mean_fit.reshape(shape_cur)
    latent_variance_fit = latent_variance_fit.reshape((*shape_cur, dim_latent))


# Color with time
color_temporal_position = torch.cat((
    torch.linspace(0, 1, model.len_observation).unsqueeze(1),
    torch.linspace(0, 0, model.len_observation).unsqueeze(1),
    torch.linspace(1, 0, model.len_observation).unsqueeze(1)),
    dim=1)

# Plot indices
plot_num = 10
plot_offset = 0
plot_index = np.arange(plot_num) + plot_offset

plt.figure(figsize=(3 * len(plot_index), 2 * 3))
for nn_id in range(len(plot_index)):

    # Current observation
    nn = plot_index[nn_id]

    # Plot true trajectory
    ax0 = plt.subplot(2, len(plot_index), 1 + nn_id + 0 * len(plot_index))
    plt.scatter(latent_true[nn, :, 0], latent_true[nn, :, 1], c=color_temporal_position, s=50, label='t=0')
    plt.title('Example ' + str(1 + nn) + '/' + str(model.num_observation))
    plt.grid()

    if nn_id == 0:
        plt.ylabel('True Trajectory \nz2[t]', multialignment='center')
    elif nn == len(plot_index)-1:
        plt.legend()

    # PLot fitted latents
    ax1 = plt.subplot(2, len(plot_index), 1 + nn_id + 1 * len(plot_index))
    latent_mean = latent_mean_fit[nn]
    latent_variance = latent_variance_fit[nn]
    plot_ellipses(latent_mean, latent_variance, ax1)
    plt.autoscale(enable=True, axis='xy', tight=True)
    plt.scatter(latent_mean[:, 0], latent_mean[:, 1], c=color_temporal_position, s=50)
    plt.xlabel('z1[t]')
    if nn_id == 0:
        plt.ylabel('Recognized Latent \nz2[t]', multialignment='center')
    plt.grid()






#%%





