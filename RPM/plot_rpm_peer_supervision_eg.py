#%% Imports
import pickle
import numpy as np
from os import listdir

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from utils import rearrange_mnist

#%%  Load MNIST and fitted recognition network
train_data = datasets.MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
    )

test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor()
    )

# Number of Conditionally independent Factors
num_factors = 2

# Train Length Used
train_length = 60000

# Digit used
sub_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Load Model
with open('./../RPM_data_final/rpm_peer_supervision_example.pkl', 'rb') as f:
    model = pickle.load(f)

# Deactivate dropouts
model.recognition_network.eval()

# Rearrange MNIST by grouping num_factors Conditionally independent Observations together
observations, train_images, train_labels = rearrange_mnist(
    train_data.train_data, train_data.train_labels, num_factors, train_length=train_length, sub_ids=sub_ids)

# Grasp Test set
test_images = test_data.test_data[torch.isin(test_data.test_labels, sub_ids)]
test_labels = test_data.test_labels[torch.isin(test_data.test_labels, sub_ids)]

# Reduce training set to find permutation
reduce_training_set = 20000

# Convert Test datasets
test_tmp = torch.tensor(test_images.clone().detach(), dtype=torch.float32)

# Use Recognition Network to classify digits
train_predictions = \
    torch.argmax(model.recognition_network.forward(train_images[:reduce_training_set].unsqueeze(dim=1)), dim=1)
test_predictions = \
    torch.argmax(model.recognition_network.forward(test_tmp.unsqueeze(dim=1)), dim=1)

# Find best permutation between model clusters and digits identity
perm_opt = model.permute_prediction(train_predictions, train_labels[:reduce_training_set], sub_ids)

# Permute Labels
train_predictions = perm_opt[train_predictions]
test_predictions = perm_opt[test_predictions]

# Train / Test performances
train_accuracy = sum(abs(train_predictions - train_labels[:reduce_training_set]) < 0.1) / reduce_training_set
test_accuracy = sum(abs(test_predictions - test_labels) < 0.01) / len(test_labels)

# Summary
train_results = str(np.round(train_accuracy.numpy(), 2))
test_results = str(np.round(test_accuracy.numpy(), 2))

print('Training Accuracy = ' + train_results)
print('Testing  Accuracy = ' + test_results)
