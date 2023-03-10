#%% Imports
import os
import torch
import pickle
import numpy as np
from utils import diagonalize, generate_skewed_pixel_from_latent
import matplotlib.pyplot as plt
from utils_process import linear_regression_1D_latent
from unstructured_recognition_gpfa import load_gprpm


#%% Load Multiple-Models

# Result Path
textured_ball_path = './../results_gp_rpm/benchmark/'
gp_rpm_path = 'gp_rpm/'
sgp_vae_path = 'sgp_vae/'

# Kernel of interest
kernel_used = 'RBFid'

# Grasp RP-GPFA Results with correct Kernel
name_tot_gprpm = os.listdir(textured_ball_path + gp_rpm_path)

# Benchmark Methods

benchmark_names = ['Samples', '2ndOrder', 'VariationalBound', 'sGPVAE1', 'sGPVAE2']
benchmark_plot_names = ['RP-GPFA Samples', 'RP-GPFA 2ndOrder', 'RP-GPFA VB', 'sGPVAE 1D', 'sGPVAE 2D']
benchmark_colors = ['tab:blue', 'lightseagreen', 'tab:green', 'tab:orange', 'tab:red']
benchmark_id = range(1, len(benchmark_names) + 1)

# Grasp names and identifiers
benchmark_files = []
benchmark_seeds = []
for method_id_cur in range(len(benchmark_names)):
    method_cur = benchmark_names[method_id_cur]
    files_cur = [name_cur for name_cur in name_tot_gprpm if (kernel_used in name_cur and method_cur in name_cur)]

    benchmark_files += [files_cur]
    if method_id_cur == 0:
        benchmark_seeds = [ff.split('id')[1].split('_')[0] for ff in files_cur]

# Consistent number of seeds
num_files = [len(nn) for nn in benchmark_files]
assert all([len(nn) == num_files[0] for nn in benchmark_files])
num_fit = num_files[0]


# Init Free Energies and R2 values
benchmark_FE = torch.zeros(len(benchmark_names), num_fit)
benchmark_R2 = torch.zeros(len(benchmark_names), num_fit)

for seed_cur in range(num_fit): # num_fit
    print('Loading ' + str(seed_cur+1) + '/' + str(num_fit))

    for method_id_cur in range(len(benchmark_names)):

            # Current Name
            current_files = benchmark_files[method_id_cur]
            current_seeds = benchmark_seeds[seed_cur]
            current_name = [nn for nn in current_files if current_seeds in nn][0]

            if not('sGPVAE' in benchmark_names[method_id_cur]):
                # Grasp GP-RPM model
                current_model, observations, _, _, true_latent = load_gprpm(textured_ball_path + gp_rpm_path + current_name)

                # Re-Estimate Free Energy
                current_FE = current_model.get_free_energy_accurate(100).detach()

                # Grasp Variational Marginals
                q = current_model.variational_marginals.suff_stat_mean
                current_mean = q[0].detach().clone()
                current_vari = diagonalize(q[1].detach().clone().diagonal(dim1=-1, dim2=-2) - current_mean ** 2)

                # Linear Regression
                _, _, _, current_R2 = linear_regression_1D_latent(true_latent, current_mean, current_vari)

                del current_model

            else:
                with open(textured_ball_path + gp_rpm_path + current_name, 'rb') as outp:
                    current_mean, current_vari, current_FE = pickle.load(outp)
                    current_FE = -current_FE[-1] * current_mean.shape[0] * current_mean.shape[1]

                    current_mean = torch.tensor(current_mean, dtype=observations[0].dtype)
                    current_vari = torch.tensor(current_vari, dtype=observations[0].dtype)

                    if benchmark_names[method_id_cur] == 'sGPVAE1':
                        current_mean = current_mean.unsqueeze(-1)
                        current_vari = current_vari.unsqueeze(-1).unsqueeze(-1)
                        _, _, _, current_R2 = linear_regression_1D_latent(true_latent, current_mean, current_vari)
                    elif benchmark_names[method_id_cur] == 'sGPVAE2':
                        _, _, _, current_R2 = linear_regression_1D_latent(true_latent, current_mean, current_vari)


            # Store R2 and Free Energy
            benchmark_FE[method_id_cur, seed_cur] = current_FE
            benchmark_R2[method_id_cur, seed_cur] = current_R2




#%%

# If FE > 0, the algorithm diverged
for seed_cur in range(num_fit):
    for method_id_cur in range(len(benchmark_names)):

        current_FE = benchmark_FE[method_id_cur, seed_cur]
        current_R2 = benchmark_R2[method_id_cur, seed_cur]

        if (current_FE > 0) or (current_FE < -1e8): #Remove cases where algorithm diverged ?
            benchmark_FE[method_id_cur, seed_cur] = float("nan")


#%% Plot Metrics
delta0 = 0.05
metrics_data = [benchmark_FE, benchmark_R2]
metrics_name = ['Free Energy', 'R2']

ylim = [[-3.5*1e4, -4.9*1e3], [0, 1]]

plt.figure(figsize=(4*(len(metrics_name)+1), 3))
for metric_id in range(len(metrics_name)):

    plt.subplot(1, len(metrics_name) + 1, metric_id +1)

    for model_id in range(len(benchmark_id)):
        yy = metrics_data[metric_id][model_id, :].detach().numpy()
        yy = yy[~np.isnan(yy)]
        cc = benchmark_colors[model_id]
        ii = benchmark_id[model_id]
        xx = np.linspace(-delta0, delta0, len(yy)) + benchmark_id[model_id]
        plt.scatter(xx, yy, c=cc, label=benchmark_plot_names[model_id])
        vp = plt.violinplot(yy, [ii], points=60, widths=0.4, showmeans=False, showextrema=True, showmedians=True,
                            bw_method=0.3)

        vp['bodies'][0].set_color(cc)
        for partname in ('cbars', 'cmaxes', 'cmins', 'cbars', 'cmedians'):
            vp[partname].set_edgecolor('k')
            vp[partname].set_linewidth(2)


        loq = np.quantile(yy, 0.25)
        hiq = np.quantile(yy, 0.75)
        med = np.quantile(yy, 0.50)

        irq = hiq - loq
        mad = np.quantile(np.abs(yy - med), 0.50)

        #print(metrics_name[metric_id] + ' = %.2e' % med + ' \pm %.2e' % irq + ' ' + benchmark_names[model_id])

        print(metrics_name[metric_id] + ' = %.1e' % med + ' [ %.1e' % loq + ' %.1e' % hiq + ' ] '  + benchmark_names[model_id])

        if metric_id == 0:
            plt.legend()


    plt.ylim(ylim[metric_id])
    plt.title(metrics_name[metric_id])
    #plt.xticks(benchmark_id, benchmark_names)
    plt.xticks([])

for model_id in range(len(benchmark_id)):
    plt.subplot(1, len(metrics_name) + 1, len(metrics_name) + 1)
    plt.scatter(metrics_data[0][model_id, :], metrics_data[1][model_id, :], c=benchmark_colors[model_id])
    plt.xlabel(metrics_name[0])
    plt.ylabel(metrics_name[1])
    plt.title(metrics_name[1] + 'Vs' + metrics_name[0])

#plt.xlim(ylim[0])


plt.xlim([-35*1e3, -4.9*1e3])
plt.ylim(ylim[1])

#%% Load 1 dataset

plot_id = benchmark_R2.sum(dim=0).argmax()


print('Loading ' + str(plot_id+1) + '/' + str(num_fit))

means = []
variances = []

for method_id_cur in range(len(benchmark_names)):

        # Current Name
        current_files = benchmark_files[method_id_cur]
        current_seeds = benchmark_seeds[seed_cur]
        current_name = [nn for nn in current_files if current_seeds in nn][0]

        if not ('sGPVAE' in benchmark_names[method_id_cur]):
            # Grasp GP-RPM ;odel
            current_model, observations, _, _, true_latent = load_gprpm(textured_ball_path + gp_rpm_path + current_name)

            # Re-Estimate Free Energy
            current_FE = current_model.get_free_energy_accurate(100).detach()

            # Grasp Variational Marginals
            q = current_model.variational_marginals.suff_stat_mean
            current_mean = q[0].detach().clone()
            current_vari = diagonalize(q[1].detach().clone().diagonal(dim1=-1, dim2=-2) - current_mean ** 2)

            # Linear Regression
            latent_true, current_mean, current_vari, _ = linear_regression_1D_latent(true_latent, current_mean, current_vari)
        else:
            with open(textured_ball_path + gp_rpm_path + current_name, 'rb') as outp:
                current_mean, current_vari, current_FE = pickle.load(outp)
                current_FE = -current_FE[-1] * current_mean.shape[0] * current_mean.shape[1]

                current_mean = torch.tensor(current_mean, dtype=observations[0].dtype)
                current_vari = torch.tensor(current_vari, dtype=observations[0].dtype)

                if benchmark_names[method_id_cur] == 'sGPVAE1':
                    current_mean = current_mean.unsqueeze(-1)
                    current_vari = current_vari.unsqueeze(-1).unsqueeze(-1)
                latent_true, current_mean, current_vari, _ = linear_regression_1D_latent(true_latent, current_mean, current_vari)



        # Store Means and Variances
        means += [current_mean]
        variances += [current_vari]

#%% Load Distribution Plot
hard_distribution = False
if hard_distribution:
    xp_name = 'hard'
    scale_th = 0.4
    shape_max_0 = 5
    sigma2 = 0.01
else:
    xp_name = 'standard'
    scale_th = 0.15
    shape_max_0 = 1000
    sigma2 = 0.01

Nsample = 100000
meanss = torch.tensor([-1, 1, - 0.64]).unsqueeze(0).repeat(Nsample, 1)
samples = generate_skewed_pixel_from_latent(meanss.unsqueeze(-1), 10, scale_th=0.15, sigma2=0.01, shape_max_0=1000)
samples = samples[:, :, 0]

plot_id = [1, 2]
plot_legend = ['far', 'close']
plot_color = ['k', [0.5, 0.5, 0.5]]

plt.figure(figsize=(4, 4))
for nn in range(len(plot_id)):
    samples_cur = samples[:, plot_id[nn]]
    plt.hist(samples_cur, bins=500, density=True, alpha = 0.75, label=plot_legend[nn], color=plot_color[nn])
    plt.hist(samples_cur, bins=500, density=True, alpha=0.75, color='k', histtype='step')
    print('Mean1: %.2e' % samples_cur.mean() + '| Var1: %.2e' % samples_cur.var())
plt.legend(title='Ball-Pixel')
plt.title('Genrative Distribution')


#%% Plot Dataset

# Obs to be plotted
plot_indices = np.arange(1) + 14

# Problem dimensions
num_observation, len_observation, dim_observation = observations[0].shape
dim_latent = 1
benchmark_plot_names = ['RP-GPFA Samples', 'RP-GPFA 2ndOrder', 'RP-GPFA VB', 'sGPVAE', 'sGPVAE']

# Methods to ne plotted
plot_method_index = [2, 3]

plt.figure(figsize=(5 * 3, 3 * (1+len(plot_indices))))
for nn_id in range(len(plot_indices)):

    nn = plot_indices[nn_id]

    # Plot All Pixels in 2D
    plt.subplot(len(plot_indices)+1, 3,  2 + nn_id * 3)
    plt.imshow(observations[0][nn].transpose(-1, -2),
               aspect='auto', cmap='gray', vmin=-1, vmax=1, extent=[0, 1, -1, 1], origin='lower')
    plt.autoscale(enable=True, axis='x', tight=True)
    #plt.ylabel('n =  ' + str(nn) + '/' + str(num_observation))
    plt.grid()
    plt.yticks([])
    if nn_id == 0:
        plt.title('Observation')
    if nn_id == (len(plot_indices) - 1):
        plt.xlabel('Time [a.u]')
        plt.xticks([0, 0.5, 1])

    pklDict = {'observations': observations[0][nn].transpose(-1, -2)}

    # Plot All Pixels counts
    plt.subplot(len(plot_indices)+1, 3, 1 + nn_id * 3)
    plot_id = [1, 2]
    plot_legend = ['far', 'close']
    plot_color = ['k', [0.5, 0.5, 0.5]]
    for nnhist in range(len(plot_id)):
        samples_cur = samples[:, plot_id[nnhist]]
        plt.hist(samples_cur, bins=500, density=True, alpha=0.75, label=plot_legend[nnhist], color=plot_color[nnhist])
        plt.hist(samples_cur, bins=500, density=True, alpha=0.75, color='k', histtype='step')
        print('Mean1: %.2e' % samples_cur.mean() + '| Var1: %.2e' % samples_cur.var())
    plt.legend(title='Ball-Pixel')
    plt.title('Generative Distribution')

    # PLot fitted latents
    plt.subplot(len(plot_indices)+1, 3,  3 + nn_id * 3)
    cmap = plt.get_cmap("tab10")
    for dim_latent_cur in range(dim_latent):

        true = latent_true[nn].squeeze()
        xx = np.arange(len(true)) / len(true)
        plt.plot(xx, true, color='k', label='True', linestyle='-.')
        pklDict['true_latent'] = true

        for method_id_cur in range(len(plot_method_index)):

            current_mean = means[plot_method_index[method_id_cur]][nn].squeeze()
            current_vari = 2 * torch.sqrt(variances[plot_method_index[method_id_cur]][nn].squeeze())

            plt.fill_between(xx, current_mean - current_vari, current_mean + current_vari, color=benchmark_colors[plot_method_index[method_id_cur]], alpha=.1)
            plt.plot(xx, current_mean, color=benchmark_colors[plot_method_index[method_id_cur]], label=benchmark_plot_names[plot_method_index[method_id_cur]], linewidth=2)

            pklDict[benchmark_plot_names[plot_method_index[method_id_cur]]+'_means'] = current_mean
            pklDict[benchmark_plot_names[plot_method_index[method_id_cur]]+'_vars'] = current_vari


    #with open(folder + 'texture_plot.pkl', 'wb') as handle:
    #    pickle.dump(pklDict, handle)


    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ylim([-1.2, 1.2])
    plt.yticks([-1, 0, 1])
    plt.grid()
    if nn_id == 0:
        plt.title('Latent')
        plt.legend(loc=1)

    if nn_id == (len(plot_indices) - 1):
        plt.xlabel('Time [a.u]')
        plt.xticks([0, 0.5, 1])










