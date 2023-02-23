import numpy as np
import matplotlib.pyplot as plt
import torch


# Size of the subsampled van Hateren Dataset
imshape = (1024, 1536)
num_obs = 99

# Store images
img_tot = torch.zeros([num_obs, *imshape])

# Load dataset
for nn in range(num_obs):

    path = './../RPM_data/dataset_images/imk00' + str(1+nn).zfill(3) + '.imc'

    with open(path, 'rb') as handle:
        s = handle.read()

    # Convert current image
    img = np.fromstring(s, dtype='uint16').byteswap()
    img = img.astype(float)

    # Normalise to [0,1]
    normalize = True
    if normalize:
        img -= img.min()
        img /= img.max()

    # Store
    img = img.reshape(imshape)
    img_tot[nn] = torch.tensor([img])

#%% Subsample and define patches

# Patches size
num_patch = 10
dim_patch = 28

# New image size
new_size = num_patch * dim_patch
down_step = np.floor(1024/new_size).astype(int)

# Init new dataset
img_tot_downsampled = torch.zeros(num_obs, new_size, new_size)
for nn in range(num_obs):
    img_tot_downsampled[nn] = img_tot[nn][range(new_size)*down_step][:, range(new_size)*down_step]

# Define patches
observations_augmented = torch.zeros(num_patch, num_patch, num_obs, dim_patch, dim_patch)
for nn in range(num_obs):
    for ii in range(num_patch):
        for jj in range(num_patch):

            xx_patch = (np.arange(dim_patch) + dim_patch*(ii)).astype(int)
            yy_patch = (np.arange(dim_patch) + dim_patch*(jj)).astype(int)

            observations_augmented[ii, jj, nn] = img_tot_downsampled[nn][xx_patch][:, yy_patch]

# Define observation as collapsed array of conditionally independent patches
observations =\
    observations_augmented.reshape(num_patch*num_patch, num_obs, dim_patch, dim_patch).permute(1, 0, 2, 3)

# Store conditionally independent patches as list
observation_list = ()
for ij in range(num_patch*num_patch):
    observation_list += (observations[:, ij, :, :],)

#%% Plot Summary of the rearranged dataset

# Image id to plot
npl = 45

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img_tot[npl], cmap="gray")
plt.axis('off')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(img_tot_downsampled[npl], cmap="gray")
plt.axis('off')
plt.title('Downsampled')

plt.figure()
for ii in range(num_patch):
    for jj in range(num_patch):
        plt.subplot(num_patch, num_patch, jj + ii*num_patch + 1)
        plt.imshow(observations[npl, jj + ii*num_patch], cmap="gray")
        plt.axis('off')

#%%
# Save Data
#  import pickle
#  with open('LDA_observations.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#      pickle.dump([observation_list, observations, img_tot], f)
