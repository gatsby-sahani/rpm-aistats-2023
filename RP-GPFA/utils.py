# Imports
import torch
import numpy as np


def print_loss(loss, epoch_id, epoch_num, pct=0.001):
    """ Simple logger"""
    str_epoch = 'Epoch ' + str(epoch_id) + '/' + str(epoch_num)
    str_loss = ' Loss: %.6e' % loss

    if epoch_num < int(1/pct) or epoch_id % int(epoch_num * pct) == 0:
        print(str_epoch + str_loss)


def check_size(observations: tuple):
    """ Check that first 2 tensors dimensions in observations are consistent """
    num_factors = len(observations)
    num_observation, len_observation = observations[0].shape[:2]
    for cur_factor in range(num_factors - 1):
        cur_num_observation, cur_len_observation = observations[cur_factor + 1].shape[:2]
        assert cur_num_observation == num_observation
        assert cur_len_observation == len_observation

    return num_factors, num_observation, len_observation


def get_minibatches(num_epoch, len_full, len_minibatch):
    """ Returns mini-batch indices """

    num_minibatch = int(np.ceil(len_full / len_minibatch))

    mini_batches = []
    for epoch in range(num_epoch):
        if len_full == len_minibatch:
            permutation_cur = list(np.arange(len_full))
        else:
            permutation_cur = list(np.random.permutation(np.arange(len_full)))
        mini_batch_cur = [np.sort(permutation_cur[i * len_minibatch:(i + 1) * len_minibatch])
                          for i in range(num_minibatch)]

        mini_batches.append(mini_batch_cur)

    return mini_batches


def minibatch_tupple(input, dim,  idx, device=None):
    """ Extract relevant minibatch from tupled multifactorial observations """
    idx = torch.tensor(idx, device=device)
    return tuple([torch.index_select(obsj, dim, idx) for obsj in input])


def optimizer_wrapper(param, optimizer_param):

    optimizer_name = optimizer_param['name']
    optimizer_param = optimizer_param['param']

    if optimizer_name == 'Adam':
        return torch.optim.Adam(param, **optimizer_param)

    elif optimizer_name == 'SGD':
        return torch.optim.SGD(param, **optimizer_param)

    elif optimizer_name == 'Adamax':
        return torch.optim.Adamax(param, **optimizer_param)

    elif optimizer_name == 'LBFGS':
        return torch.optim.LBFGS(param, **optimizer_param)

    elif optimizer_name == 'RMSprop':
        return torch.optim.RMSprop(param, **optimizer_param)

    elif optimizer_name == 'AdamW':
        return torch.optim.AdamW(param, **optimizer_param)

    else:
        raise NotImplementedError()


def diagonalize(z):
    """ Use a batch vector to create diagonal batch matrices """
    Z = torch.zeros((*z.shape, z.shape[-1]), device=z.device, dtype=z.dtype)
    Z[..., range(z.shape[-1]), range(z.shape[-1])] = z
    return Z







