import random
import numpy as np

import torch
from utils import reshape_fortran


def get_reduced_coordinate(chars, color_background):
    # Background
    background_tiles = (chars == color_background)

    # Identify Background
    xmin = np.min((1 - background_tiles).nonzero()[0])
    xmax = np.max((1 - background_tiles).nonzero()[0])

    ymin = np.min((1 - background_tiles).nonzero()[1])
    ymax = np.max((1 - background_tiles).nonzero()[1])

    coordinate = (xmin, xmax, ymin, ymax)

    return coordinate, background_tiles

def get_transition_matrix_empirical(state_ids, trajectory):
    # Empirical marginals
    state_num = (trajectory.unsqueeze(0) == state_ids.unsqueeze(1)).sum(1)
    num_state = len(state_ids)

    trajectory_length = len(trajectory)

    # Empirical transitions
    transition_matrix_empirical = torch.zeros(num_state, num_state)
    for dig in range(num_state):
        idx = ((trajectory == state_ids[dig]).nonzero())
        idx = idx[idx != (trajectory_length - 1)]
        transition_counts = (trajectory[idx + 1].unsqueeze(1) == state_ids.unsqueeze(0)).sum(0)
        transition_counts = transition_counts / (torch.sum(transition_counts) + 1e-7)
        transition_matrix_empirical[dig] = transition_counts

    return transition_matrix_empirical, state_num



def reduced_to_full_maze(reduced, full_shape, xmin, xmax, ymin, ymax):
    full_size = torch.zeros(full_shape)
    full_size[xmin:xmax + 1][:, ymin:ymax + 1] = reduced
    return full_size


def full_to_reduced_maze(full,  xmin, xmax, ymin, ymax):
    return full[xmin:xmax + 1][:, ymin:ymax + 1]


def make_step(m, a, k, xx,yy):
    for i in range(xx):
        for j in range(yy):
            if m[i][j] == k:
                if i > 0 and m[i-1][j] == 0 and a[i-1][j] == 0:
                    m[i-1][j] = k + 1
                if j > 0 and m[i][j-1] == 0 and a[i][j-1] == 0:
                    m[i][j-1] = k + 1
                if i < len(m)-1 and m[i+1][j] == 0 and a[i+1][j] == 0:
                    m[i+1][j] = k + 1
                if j < len(m[i])-1 and m[i][j+1] == 0 and a[i][j+1] == 0:
                    m[i][j+1] = k + 1
    return m


def get_maze_distance(x0, walkable, ite=200):

    dist = torch.zeros(walkable.shape)
    dist[x0[0], x0[1]] = 1

    for kk in range(ite):
        dist = make_step(dist, 1-walkable.astype(int), 1+kk, dist.shape[0] ,dist.shape[1])
    return dist


def get_maze_transition(x0, walkable, lenv=10):
    dist = get_maze_distance(x0, walkable, ite=100)

    if lenv > 0:
        dist = torch.exp(-dist/lenv)*walkable
        dist = dist / torch.sum(dist)

    return dist


def reduced_to_state(mat, walkable):
    state_index = get_states(walkable)
    return mat[state_index[:, 0], state_index[:, 1]]


def state_to_reduced(red, walkable):
    state_index = get_states(walkable)
    mat = torch.zeros(walkable.shape)
    mat[state_index[:, 0], state_index[:, 1]] = red
    return mat


def get_states(walkable):
    xx, yy = walkable.shape

    matrix_index = reshape_fortran(torch.arange(0, xx * yy, 1), (xx, yy))
    return (matrix_index * walkable).nonzero()


def get_transition_matrices(walkable, lenv=10):

    # Shape of the maze
    xx, yy = walkable.shape

    # Number of reachable states
    num_state = walkable.sum()

    # Full transition matrix
    transition_full = torch.zeros(xx * yy, xx * yy)

    # Compact transition (no unreachable)
    transition_comp = torch.zeros(num_state, num_state)

    # Reachable states 2D index
    state_idx = get_states(walkable)

    # Reachable states linear index
    linear_index = torch.arange(xx * yy).reshape(xx, yy)
    linear_index = linear_index[state_idx[:, 0], state_idx[:, 1]]

    # Loop over reachable states
    for cur_state in range(num_state):
        cur_state_index = state_idx[cur_state]
        transition_mat = get_maze_transition(cur_state_index, walkable, lenv=lenv)
        transition_vec = reduced_to_state(transition_mat, walkable)

        transition_comp[cur_state] = transition_vec
        transition_full[linear_index[cur_state]] = transition_mat.flatten()

    return transition_full, transition_comp


def get_distance(walkable, lenv=10):

    # Shape of the maze
    xx, yy = walkable.shape

    # Number of reachable states
    num_state = walkable.sum()

    # Full transition matrix
    transition_full = torch.zeros(xx, yy, xx, yy)

    # Reachable states 2D index
    state_idx = get_states(walkable)


    # Loop over reachable states
    for cur_state in range(num_state):
        cur_state_index = state_idx[cur_state]
        transition_mat = get_maze_transition(cur_state_index, walkable, lenv=lenv)

        transition_full[state_idx[cur_state, 0], state_idx[cur_state, 1], :, :] = transition_mat

    return transition_full


def get_reduced_coord(idx, coordinate):
    idx_new = idx
    idx_new[0] = idx_new[0] - coordinate[0]
    idx_new[1] = idx_new[1] - coordinate[2]
    return idx_new


def string_to_command(strs):
    if strs == 'up':
        cmd = 0
    elif strs == 'right':
        cmd = 1
    elif strs == 'down':
        cmd = 2
    elif strs == 'left':
        cmd = 3

    return cmd


def get_path(dist, start, stop):
    # 0: up
    # 1: right
    # 2: down
    # 3: left

    m = dist[start[0],start[1]].clone()

    i, j = stop
    k = m[i][j]
    the_path = torch.zeros(int(k), 2)
    the_path[int(k)-1] = torch.tensor([i, j])
    the_step = ()

    while k > 1:
        if i > 0 and m[i - 1][j] == k-1:
            i, j = i-1, j
            the_step = the_step + ('down',)

        elif j > 0 and m[i][j - 1] == k-1:
            i, j = i, j-1
            the_step = the_step + ('right',)

        elif i < len(m) - 1 and m[i + 1][j] == k-1:
            i, j = i+1, j
            the_step = the_step + ('up',)

        elif j < len(m[i]) - 1 and m[i][j + 1] == k-1:
            i, j = i, j+1
            the_step = the_step + ('left',)

        k -= 1
        the_path[int(k)-1] = torch.tensor([i, j])

    the_steps_action = path_to_step(the_step)

    return the_path, the_steps_action, the_step


def path_to_step(path):
    steps = torch.zeros(len(path))
    for ii in range(len(path)):
        steps[ii] = string_to_command(path[ii])

    return (steps)


def contaminate_obs(obs_true, noise_model, param):
    visual_field = obs_true.shape
    if noise_model =='gaussian':
        obs_noisy =  obs_true + torch.sqrt(torch.tensor([param])) * torch.randn(visual_field)

    elif noise_model=='bernoulli':
        obs_noisy = obs_true*torch.bernoulli((1-param)*torch.ones(visual_field))

    elif noise_model == 'jitter':
        old_shape = obs_true.shape
        new_shape = param

        dx_max = old_shape[0] - new_shape
        dy_max = old_shape[1] - new_shape

        dx = random.choice(range(dx_max))
        dy = random.choice(range(dy_max))
        obs_noisy = obs_true[dx:(dx + new_shape), dy:(dy + new_shape)]

    return obs_noisy

def reduced_coordinate_to_state_id(state_idx, coord):
    return (torch.sum((state_idx - coord)**2, dim=1)==0).nonzero().squeeze().int()
