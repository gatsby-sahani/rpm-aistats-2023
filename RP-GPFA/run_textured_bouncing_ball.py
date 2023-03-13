import torch
import numpy as np
import matplotlib.pyplot as plt
from utils_generate_toydatasets import generate_2D_latent, generate_skewed_pixel_from_latent
from recognition_parametrised_gpfa import RPGPFA

# Reproducibility
np.random.seed(1)
torch.manual_seed(1)

# GPUs ?
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dimension of the observations
num_observation = 20
dim_observation = 20
len_observation = 50
num_inducing = 20

# Oscillation Speed
omega = 0.5

# Oscillations Parameters
F = 10  # Sampling Frequency [Hz]
T = int(len_observation / F)  # Sample Length [sec]

# Random initializations and Oscillation
theta = 2*np.pi*np.random.rand(num_observation)
z0 = torch.tensor(np.array([np.cos(theta), np.sin(theta)]).T)
zt, _ = generate_2D_latent(T, F, omega, z0)

# Distribution parameters
scale_th = 0.15
shape_max_0 = 1000
sigma2 = 0.01

# True Latent
latent_true = zt[:, 1:, 0] .unsqueeze(-1)

# Sample Observations
samples = generate_skewed_pixel_from_latent(latent_true, dim_observation, scale_th=scale_th, sigma2=sigma2)

# Convert Observations
observations = torch.tensor(samples, dtype=dtype, device=device)

#%% Plot Typical Sample

# Sample from distribution
N = 100000
latent_test = torch.empty(N, 3, 1)
latent_test[:, 0, 0] = 1
latent_test[:, 1, 0] = 0
latent_test[:, 2, 0] = -1
observation_samples = generate_skewed_pixel_from_latent(latent_test, 30, scale_th=scale_th, sigma2=sigma2)

# Plot Distribution
plt.figure(figsize=(6 * 2, 4))
plt.subplot(1, 2, 1)
s1 = observation_samples[:, 0, 0]
s2 = observation_samples[:, 0, 24]
plt.hist(s1, bins=500, density=True, alpha=0.75, color=[0.00, 0.00, 0.00], label='Far')
plt.hist(s2, bins=500, density=True, alpha=0.75, color=[0.25, 0.25, 0.75], label='Close')
plt.xlabel('Pixel Intensity')
plt.title('Intensity Distribution')
plt.legend(title='Ball distance')
print('Mean1: %.2e' % s1.mean() + '| Var1: %.2e' % s1.var())
print('Mean2: %.2e' % s2.mean() + '| Var2: %.2e' % s2.var())

# Plot Typical Observation
egobs = 0
pixel_plot = [0, int(dim_observation / 4), 4 * int(dim_observation / 4) -1  ]
plot_observations = not(torch.cuda.is_available())
plt.subplot(1, 2, 2)
plt.imshow(samples[0].transpose(-1, -2), aspect='auto', cmap='gray', extent=[0, 1, -1, 1], vmin=-1, vmax=1)
plt.title('Observation ' + str(egobs) + '/' + str(num_observation))
plt.ylabel('Pixel Id.')
plt.xlabel('Time [a.u]')
plt.tight_layout()



#%% Fit and Save

# Set up observation / inducing locations
inducing_locations = torch.linspace(0, 1, num_inducing, dtype=dtype, device=device).unsqueeze(-1)
observation_locations = torch.linspace(0, 1, len_observation, dtype=dtype, device=device).unsqueeze(-1)
observations = (observations,)


# Linear / Non Linear Network
import torch.nn.functional as F
linear_networks = False
dim_hidden0 = [] if linear_networks else [10, 10]
non_lineraity0 = torch.nn.Identity() if linear_networks else F.relu
num_factors = 1

# Copy for each factors
dim_hidden = tuple([dim_hidden0 for _ in range(num_factors)])
neural_net = tuple(['perceptron' for _ in range(num_factors)])
nonlinearity = tuple([non_lineraity0 for _ in range(num_factors)])

fit_params = {'dim_latent': 1,
              'constraint_factors': 'fixed_diag',
              'num_epoch': 2500,
              'optimizer_prior': {'name': 'Adam', 'param': {'lr': 1e-3}},
              'optimizer_factors': {'name': 'Adam', 'param': {'lr': 1e-3}},
              'optimizer_inducing_points': {'name': 'Adam', 'param': {'lr': 1e-3}},
              'gp_kernel': 'RBF',
              'dim_hidden': dim_hidden,
              'non_linearity':nonlinearity,
              'nn_type': neural_net,
              'fit_prior_mean': True}

# Init Model
model = RPGPFA(observations, observation_locations,
               inducing_locations=inducing_locations, fit_params=fit_params)

# Fit Model
loss_tot = model.fit(observations)

#%%




#plot_summary(model, observations, true_latent=true_latent_ext, offset=0, plot_n = 3)

#%%
from utils_process import plot_summary, plot_factors_prior, plot_loss


plot_loss(model)
plot_factors_prior(model)
plot_summary(model, latent_true=latent_true, plot_observation=[0], plot_factors_id='all', plot_regressed=False)



