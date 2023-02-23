#%% Imports
import torch
import pickle
from matplotlib import pyplot as plt

from utils import inv_perm
from unstructured_recognition_MazeCategorical import permute_prediction

#%% Load HMM-MNIST and Handle possible state permutations
with open('./../RPM_data_final/rpm_hmm_mnist.pkl', 'rb') as f:
    model, trajectory_image, trajectory, sub_ids, loss_tot, transition_matrix_true = pickle.load(f)

# Model fits
latent_marginals, _ = model.latent
transition_matrix = model.transition_matrix

# Get variational-MAP trajectory
most_likely_trajectory = torch.argmax(latent_marginals, dim=1)

# Get Best State Permutation
perm = permute_prediction(most_likely_trajectory, trajectory, sub_ids)

# Permute Trajectory
most_likely_trajectory = perm[most_likely_trajectory]

# Permute Transition Matrix
transition_matrix_perm = transition_matrix[inv_perm(perm)][:, inv_perm(perm)]

#%% Plot HMM-MNIST

plt.figure(figsize=(10, 10))

# Loss
plt.subplot(2, 2, 1)
plt.plot(loss_tot, color='k')
plt.xlabel('Iteration')
plt.title('- Free Energy')

# Transition True
plt.subplot(2, 2, 3)
plt.imshow(transition_matrix_true)
plt.title('True Transition')
plt.axis('off')

# Transition Fit
plt.subplot(2, 2, 4)
plt.title('Fit Transition')
plt.imshow(transition_matrix_perm.detach().numpy())
plt.axis('off')

# True and Fit Trajectories
plt.subplot(2, 2, 2)
plt.plot(trajectory, label='True', color='k', linewidth=4)
plt.plot(most_likely_trajectory, label='MAP', color='orange', linestyle='dashed', linewidth=3)
plt.xlim([0, 500])
plt.title('Latent Trajectory')
plt.xlabel('Step Number')
plt.ylabel('Digit Id')


#%% Load HMM-Minihack and Handle possible state permutations
with open('./../RPM_data_final/rpm_hmm_minihack_maze.pkl', 'rb') as f:
    model, transition_comp, trajectory = pickle.load(f)

# Model fits
latent_marginals, _ = model.latent
transition_matrix = model.transition_matrix

# Get variational-MAP trajectory
most_likely_trajectory = torch.argmax(latent_marginals, dim=1)

# Get Best State Permutation
perm = permute_prediction(most_likely_trajectory, trajectory, torch.arange(model.num_state))

# Permute Trajectory
transition_matrix_true = transition_comp
trajectory = torch.tensor([inv_perm(perm)]).squeeze()[trajectory.long()]

# Permute Transition Matrix
transition_matrix_perm = transition_matrix[inv_perm(perm)][:, inv_perm(perm)]

#%% Plot HMM-Minihack
plt.figure(figsize=(10, 10))

# Loss
plt.subplot(2, 2, 1)
plt.plot(model.loss_tot, color='k')
plt.xlabel('Iteration')
plt.title('- Free Energy')

# Transition True
plt.subplot(2, 2, 3)
plt.imshow(transition_matrix_true, vmin=transition_matrix_true.min(), vmax=transition_matrix_true.max())
plt.title('True Transition')
plt.axis('off')

# Transition Fit
plt.subplot(2, 2, 4)
plt.title('Fit Transition')
plt.imshow(transition_matrix_perm.detach().numpy(), vmin=transition_matrix_true.min(), vmax=transition_matrix_true.max())
plt.axis('off')

# True and Fit Trajectories
plt.subplot(2, 2, 2)
plt.plot(trajectory, label='True', color='k', linewidth=4)
plt.plot(most_likely_trajectory, label='VMAP', color='orange', linestyle='dashed', linewidth=3)
plt.xlim([8500, 8600])

plt.title('Latent Trajectory')
plt.xlabel('Step Number')
plt.ylabel('Digit Id')
plt.legend()


#%%

'''
import numpy as np
# True and Fit Trajectories
plt.figure(figsize=(3, 2))
x1 = 100+8500
x2 = 100+8600
plt.plot(np.arange(x2-x1), trajectory[x1:x2], label='True', color='k', linewidth=4)
plt.plot(np.arange(x2-x1),most_likely_trajectory[x1:x2], label='VMAP', color='orange', linestyle='dashed', linewidth=3)

#plt.title('Latent Trajectory')
plt.xlabel('Step Number')
plt.ylabel('State Id')
plt.legend()

# True and Fit Trajectories
plt.figure(figsize=(3, 2))
x1 = 8500
x2 = 8600
plt.plot(np.arange(x2-x1), trajectory[x1:x2], label='True', color='k', linewidth=4)
plt.plot(np.arange(x2-x1),most_likely_trajectory[x1:x2], label='VMAP', color='orange', linestyle='dashed', linewidth=3)

#plt.title('Latent Trajectory')
plt.xlabel('Step Number')
plt.ylabel('State Id')
plt.legend()
'''

