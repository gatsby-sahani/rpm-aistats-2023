# Imports
import torch
import numpy as np
from torch import exp
from torch import matmul
from torch.nn.functional import softplus
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import norm, bernoulli
import soundfile as sf
from pathlib import Path


def gp_kernel(name: str):

    def square_distance(locations1, locations2):
        # Distances between locations
        # locations1 ~ N1 x D
        # locations2 ~ N2 x D

        # locations1 - locations2 ~ N1 x N2 x D
        diff = locations1.unsqueeze(-2) - locations2.unsqueeze(-3)

        return matmul(diff.unsqueeze(-2), diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    if name == 'RBF':
        # Exponentiated quadratic kernel

        def batch_kernel(locations1, locations2, *args):

            # ||locations1 - locations2||^2 ~ 1 x N1 x N2
            sdist = square_distance(locations1, locations2).unsqueeze(0)

            # RBF Parameters for each independent dimensions
            sigma1, scale1 = args[0]

            # Expand and square
            sigma21_expanded = (sigma1 ** 2).unsqueeze(-1).unsqueeze(-1)
            scale21_expanded = (scale1 ** 2).unsqueeze(-1).unsqueeze(-1)

            # K(locations1, locations2)
            K = sigma21_expanded * torch.exp(- 0.5 * sdist / scale21_expanded)

            return K

    if name == 'RBFn':
        # Noisy Exponentiated quadratic kernel with noise

        def batch_kernel(locations1, locations2, *args):

            # ||locations1 - locations2||^2 ~ 1 x N1 x N2
            sdist = square_distance(locations1, locations2).unsqueeze(0)

            # RBF Parameters for each independent dimensions
            sigma1, scale1, sigma0 = args[0]

            # Expand and square
            sigma21_expanded = (sigma1 ** 2).unsqueeze(-1).unsqueeze(-1)
            scale21_expanded = (scale1 ** 2).unsqueeze(-1).unsqueeze(-1)
            sigma20_expanded = (sigma0 ** 2).unsqueeze(-1).unsqueeze(-1)

            # K(locations1, locations2)
            K = sigma20_expanded * (sdist == 0) + sigma21_expanded * torch.exp(- 0.5 * sdist / scale21_expanded)

            return K

    if name == 'RQ':
        # Rational quadratic kernel
        def batch_kernel(locations1, locations2, *args):

            # ||locations1 - locations2||^2 ~ 1 x N1 x N2
            sdist = square_distance(locations1, locations2).unsqueeze(0)

            # RQ Parameters for each independent dimensions
            sigma, scale, alpha = args[0]

            # Expand and square
            sigma_expanded = (sigma ** 2).unsqueeze(-1).unsqueeze(-1)
            scale_expanded = (scale ** 2).unsqueeze(-1).unsqueeze(-1)
            alpha_expanded = (alpha ** 2).unsqueeze(-1).unsqueeze(-1)

            # K(locations1, locations2)
            K = sigma_expanded * (1 + sdist / (2 * alpha_expanded * scale_expanded)) ** (-alpha_expanded)

            return K

    if 'Matern' in name:
        # Grasp order (eg. Matern52 -> nu = 5)
        nu = int(name.split('Matern')[1][0])

        # Matern Kernel nu/2
        def batch_kernel(locations1, locations2, *args):

            # ||locations1 - locations2||^2 ~ 1 x N1 x N2
            sdist = square_distance(locations1, locations2).unsqueeze(0)

            # RQ Parameters for each independent dimensions
            sigma, scale = args[0]

            # Expand and square
            sigma_expanded = (sigma ** 2).unsqueeze(-1).unsqueeze(-1)
            scale_expanded = (scale ** 2).unsqueeze(-1).unsqueeze(-1)

            # Exponential Term
            expd = torch.exp(-torch.sqrt(sdist * nu) / scale_expanded)

            # Order dependent term
            if nu == 1:
                cons = 1
            elif nu == 3:
                cons = 1 + torch.sqrt(sdist * nu) / scale_expanded
            elif nu == 5:
                cons = 1 + torch.sqrt(sdist * nu) / scale_expanded + sdist * nu / (3 * scale_expanded**2)

            # K(locations1, locations2)
            K = sigma_expanded * cons * expd

            return K

    if name == 'Periodic':
        def batch_kernel(locations1, locations2, *args):
            # ||locations1 - locations2||^2 ~ 1 x N1 x N2
            dist = torch.sqrt(square_distance(locations1, locations2)).unsqueeze(0)

            # RQ Parameters for each independent dimensions
            sigma, scale, period = args[0]

            # Expand and square
            sigma_expanded = (sigma ** 2).unsqueeze(-1).unsqueeze(-1)
            scale_expanded = (scale ** 2).unsqueeze(-1).unsqueeze(-1)
            period_expanded = (period ** 2).unsqueeze(-1).unsqueeze(-1)

            K = sigma_expanded * torch.exp(- 2 * torch.sin(dist * np.pi / period_expanded) ** 2 / scale_expanded)

            return K

    return batch_kernel


def optimizer_wrapper(params, name: str, **kwargs):
    if name == 'Adam':
        return torch.optim.Adam(params, **kwargs)

    elif name == 'SGD':
        return torch.optim.SGD(params, **kwargs)

    elif name == 'Adamax':
        return torch.optim.Adamax(params, **kwargs)

    elif name == 'LBFGS':
        return torch.optim.LBFGS(params, **kwargs)

    elif name == 'RMSprop':
        return torch.optim.RMSprop(params, **kwargs)

    elif name == 'AdamW':
        return torch.optim.RMSprop(params, **kwargs)


    else:
        raise NotImplementedError()



def print_loss(ite, ite_max, loss_cur, sub_step=None):
    print('Iterations ' + str(ite) + '/' + str(ite_max) + ' Loss: %.6e' % loss_cur + ' ' + sub_step )


def soft_bound(x, bound=0, beta=1, mode='upper'):
    if mode == 'upper':
        return bound - softplus(- (x-bound), beta=beta)
    elif mode == 'lower':
        return bound + softplus(x-bound, beta=beta)
    else:
        raise NotImplementedError()


def diagonalize(z):
    Z = torch.zeros((*z.shape, z.shape[-1]), device=z.device, dtype=z.dtype)
    Z[..., range(z.shape[-1]), range(z.shape[-1])] = z
    return Z


def threshold_eigh(batch_matrices, bound=-1e-6, beta=10, mode='upper', jitter=True):

    dtype = batch_matrices.dtype
    device = batch_matrices.device

    # Orthogonal eigen decompositions
    (eigenvalues, eigenvectors) = torch.linalg.eigh(batch_matrices)

    # Soft threshold eigenvalues
    thresholded_eigenvalues = soft_bound(eigenvalues, bound=bound, beta=beta, mode=mode)

    # Add Some noise to make eigenvalue distinct
    if jitter:
        thresholded_eigenvalues -= 1e-3 * (eigenvalues > bound) * torch.rand(eigenvalues.shape, dtype=dtype, device=device)

    # Reconstruct Matrices
    thresholded_matrices = \
        matmul(eigenvectors, matmul(diagonalize(thresholded_eigenvalues), eigenvectors.transpose(-1, -2)))

    return thresholded_matrices


def generate_lorents(state, t):
    # Lorentz Attractor
    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0

    x, y, z = state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives


def generate_2D_latent(T, F, omega, z0, noise=0.0):
    # Generate a 2D oscillation

    # Number of Time point
    L = T * F

    # Number of trajectories
    N = z0.shape[0]

    # Time Vector
    t = np.arange(L) / F

    # Rotation angle
    Omega = torch.tensor([2*np.pi * omega / F])

    # Rotation Matrix
    rotation = torch.tensor(
        [[torch.cos(Omega), -torch.sin(Omega)],
         [torch.sin(Omega), torch.cos(Omega)]])
    zt = torch.zeros(N, L + 1, 2)

    noise_mvn = MultivariateNormal(torch.zeros(2),
                                   (noise+1e-20) * torch.eye(2))

    # Loop over init
    for n in range(N):

        # Init
        zt[n, 0] = z0[n]

        # Loop Over time point
        for tt in range(L):
            zc = zt[n, tt]
            zz = torch.matmul(rotation, zc)

            if noise>0:
                zz += 0*noise_mvn.sample()

            zt[n, tt+1] = zz

    return zt, t


def smooth_bound(max_val, a=1e-3):
    def f(x):
        sigm = 1 - exp(- a / (-x+max_val)**2) * torch.gt(-x, -max_val)
        return max_val * sigm + x * (1 - sigm)
    return f


def sharp_bound(max_val):
    def f(x):
        return max_val - torch.nn.ReLU()(max_val - x)
    return f


def generate_skewed_pixel_from_latent(true_latent, dim_observation, scale_th=0.15, sigma2=0.01, shape_max_0=1000):

    # Max and min value of the latent
    latent_max = true_latent.max()
    latent_min = true_latent.min()

    # Build Observation from 1st latent
    pixel_loc = torch.linspace(latent_min, latent_max, dim_observation).unsqueeze(0).unsqueeze(0)

    # Distance Pixel - Ball
    distance_pixel_ball = (torch.exp(-(pixel_loc - true_latent) ** 2 / scale_th ** 2)).numpy()

    # From Rate to shape parameter
    shape_min = np.sqrt(1 - sigma2)
    shape_max = shape_max_0 - shape_min
    shape_parameter = shape_max * distance_pixel_ball + shape_min

    # From shape to samples
    loc0 = shape_parameter
    var0 = np.ones(shape_parameter.shape) * sigma2

    # Bernouilli Parameter
    ber0 = (1 - var0) / (1 - var0 + loc0 ** 2)

    # Mean of the First Peak
    loc1 = loc0

    # Mean of The Second Peak
    loc2 = - loc1 * ber0 / (1 - ber0)

    # Bernouilli Sample
    pp = bernoulli.rvs(ber0)

    # Assign to one distribution
    loc_cur = pp * loc1 + (1 - pp) * loc2

    # Sample feom the distribution
    observation_samples = norm.rvs(loc_cur, np.sqrt(var0))

    return observation_samples


def cuboid_data(o, size=(1, 1, 1)):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1], o[1], o[1]],
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]
    z = [[o[2], o[2], o[2], o[2], o[2]],
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]
    return np.array(x), np.array(y), np.array(z)


def plotCube(pos=(0, 0, 0), size=(1, 1, 1), ax=None, **kwargs):
    # Plotting a cube element at position pos
    if ax !=None:
        X, Y, Z = cuboid_data(pos, size)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, **kwargs)


def plotSphere(ax, position, radius, color):

    # Make data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = position[0] + radius[0] * np.outer(np.cos(u), np.sin(v))
    y = position[1] + radius[1] * np.outer(np.sin(u), np.sin(v))
    z = position[2] + radius[2] * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(x, y, z, color=color, linewidth=0)

def fast_XDXT(X, D):
    return matmul(X * D.unsqueeze(-2), X.transpose(-1, -2))




def get_speech_samples(distance_from_fixed, audio_path, downsample=10, len_snippet=1000, normalize=True):

    num_observations, len_observation, _ = distance_from_fixed.shape
    data_type = distance_from_fixed.dtype
    device = distance_from_fixed.device

    speech_snips = np.zeros((num_observations, len_observation, len_snippet))

    cur_obs = 0
    cur_time = 0

    # Grasp all audio files
    for path in Path(audio_path).rglob('*.wav'):

        # Grasp data
        data, samplerate = sf.read(path)

        # Downsample data
        data = data[::downsample]

        # Get the length (in second) of each audio snippet
        samplerate = samplerate / downsample
        time_snippet_sec = len_snippet / samplerate


        # Number of snippet in the current data file
        snippet_num_cur = int(np.floor(len(data)/len_snippet))

        for inds in range(snippet_num_cur):

            # Break if filled all observations with sound snippet
            if cur_obs >= num_observations:
                break

            # Fill Current observation and current time with sound snippet
            speech_snips[cur_obs, cur_time] = data[len_snippet * inds:len_snippet * (inds + 1)]

            # Update Time
            cur_time += 1

            # If all time point have been filled, move to next observation
            if cur_time >= len_observation:
                cur_obs += 1
                cur_time = 0

    # Error if you didn't fill the array
    if cur_obs < num_observations:
        raise Exception('Not enough speech snippets in timit')

    # Get speech samples
    speech_data = torch.tensor(speech_snips, dtype=data_type, device=device)
    if normalize:
        speech_data = speech_data - speech_data.mean(dim=-1, keepdim=True)
        speech_data = speech_data / speech_data.max(dim=-1)[0].unsqueeze(-1)

    # Maximum distance in the square arena
    dmax = torch.sqrt(torch.tensor([8]))

    # Enveloppe signal
    modulation = (dmax - distance_from_fixed) / dmax

    # Modulate
    distance_modulated_speech = (modulation.unsqueeze(-1) * speech_data.unsqueeze(-2))

    return distance_modulated_speech, time_snippet_sec







