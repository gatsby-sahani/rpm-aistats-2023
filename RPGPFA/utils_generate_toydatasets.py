# Imports
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import norm, bernoulli


def generate_lorenz(num_cond, num_steps, dt=0.01, init_simulation = [2.3274,  3.8649, 18.2295], vari_simulation=0.5):
    xyzs = np.empty((num_cond, num_steps + 1, 3))  # Need one more for the initial values

    for n in range(num_cond):
        xyzs[n, 0] = 1*np.random.randn(3)  # Set initial values

        xyzs[n, 0] = 1 * np.array(init_simulation) + vari_simulation * np.random.randn(3)

        for i in range(num_steps):
            xyzs[n, i + 1] = xyzs[n, i] + lorenzz(xyzs[n, i]) * dt

    return xyzs


def lorenzz(xyz, *, s=10, r=28, b=2.667):

    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z

    return np.array([x_dot, y_dot, z_dot])


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


