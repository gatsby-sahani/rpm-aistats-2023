# Imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle

# Load image dataset
with open('./../RPM_data/LDA/LDA_observations.pkl', 'rb') as f:
    [observation_list, observations, img_tot] = pickle.load(f)

# Load Model
with open('./../RPM_data/LDA/LDA_RPM.pkl', 'rb') as f:
    model = pickle.load(f)

# Dimensions of the problem
num_obs, num_factors, num_textures = model.latent[1].probs.shape
dim_patch = observations.shape[-1]


#%% Plot Loss
plt.figure()
plt.plot(model.loss_tot, color='k')
plt.xlabel('Iteration')
plt.title('- Free Energy')

#%% Get each image textural statistics
posterior_dirichlet_param = model.latent[0]
posterior_dirichlet = posterior_dirichlet_param / torch.sum(posterior_dirichlet_param, dim=-1, keepdim=True)

#%% Plot Images texture distribution
plt.figure(figsize=(10, 2))
plt.imshow(posterior_dirichlet.detach().numpy().transpose(), vmin=0, vmax=1)
plt.xlabel('Image Id')
plt.ylabel('Texture Id')
plt.title('Image textural Distribution')

#%% Plot texture distribution for one image
implot = 43
plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.plot(posterior_dirichlet[implot], color='k')
plt.xlabel('Texture Id')
plt.ylabel('theta^n')

plt.subplot(1, 2, 2)
plt.imshow(img_tot[implot], cmap="gray", vmin=0, vmax=0.5)
plt.axis('off')

#%% Select and plot most representative Patches

# Score the textures
texture_score = posterior_dirichlet.sum(0).detach().numpy()
texture_order = np.argsort(texture_score)[::-1]
texture_order = np.arange(num_textures).astype(int)

# Recognition Factors
recognition_probs = [factor.probs.unsqueeze(1) for  factor in model.factor_indpt]
recognition_probs = torch.cat(recognition_probs, dim=1)
#recognition_probs = model.latent[1].probs

KMAX = 10
Neigen = 5
eigen_patches = torch.zeros(num_textures, Neigen, dim_patch, dim_patch)
for k in range(num_textures):
    # Prob associated with z = k
    # what are the top Neigen Patches ?
    factor_k = recognition_probs[:, :, k].reshape(-1)
    idx = factor_k.argsort()
    idx2 = idx[-Neigen:].numpy()
    idx2 = torch.tensor([idx2[::-1]])

    idx_k = torch.cat([torch.tensor(np.unravel_index(idx2, recognition_probs[:, :, k].shape, 'C'))]).squeeze()

    for eigen_cur in range(Neigen):
        idx_ke = idx_k[:, eigen_cur]
        num_obs_cur = idx_ke[0]
        num_pat_cur = idx_ke[1]

        patch_cur = observations[num_obs_cur, num_pat_cur]
        eigen_patches[k, eigen_cur] = patch_cur

plt.figure(figsize=(1*KMAX, 1*Neigen))
for k in range(KMAX):
    for eigen_cur in range(Neigen):

        kcur = texture_order[k]

        plt.subplot(Neigen, KMAX, k + eigen_cur*KMAX+1)

        if np.isin(kcur, [1]):
            vmin = 0
            vmax = 0.8
            plt.imshow(eigen_patches[kcur, eigen_cur], cmap='gray', vmin=vmin, vmax=vmax)
        else:
            vmin = 0
            vmax = 1
            plt.imshow(eigen_patches[kcur, eigen_cur], cmap='gray')

        #, cmap="gray"
        plt.axis('off')

        if eigen_cur == 0:
            plt.title('k = ' + str(kcur))

#%% Plot typical images for each texture

kplot_tot = [0, 1, 2, 3, 5, 6]
imtt = 4

plt.figure(figsize=(len(kplot_tot)*3,imtt*2 ))
for kid in range(len(kplot_tot)):

    kcur = kplot_tot[kid]
    max_im = posterior_dirichlet[:, kcur].argsort()
    max_im = max_im[-imtt:]
    imbest = img_tot[max_im]

    for im in range(imtt):
        imc = imtt - im -1
        imc = im
        plt.subplot(imtt, len(kplot_tot), 1 + kid + im * len(kplot_tot))
        if np.isin(kid, [ 2, 6]):
            plt.imshow(imbest[imc], vmin=0, vmax=0.2, cmap='gray')
        else:
            plt.imshow(imbest[imc], vmin=0,vmax=0.7,cmap='gray')
        plt.axis('off')

        if im==0:
            #plt.axis('on')
            plt.title('k=' + str(kcur))


