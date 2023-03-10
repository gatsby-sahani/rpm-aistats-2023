import pickle
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from utils import rearrange_mnist
from unstructured_recognition_1DCategorical import UnstructuredRecognition

# Load MNIST
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

# Random seeds
torch.manual_seed(1)

# Number of Conditionally independent Factors
num_factors = 2

# Sub-Sample original dataset
train_length = 60000

# Keep Only some digits
sub_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
num_digits = len(sub_ids)

# Rearrange MNIST by grouping num_factors Conditionally independent Observations together
observations, train_images, train_labels = rearrange_mnist(
    train_data.train_data, train_data.train_labels, num_factors, train_length=train_length, sub_ids=sub_ids)


#%% Plot a quick illustration of how images are grouped together
num_plot = np.arange(6)
plt.figure(figsize=(len(num_plot)*2, num_factors*2))
for obsi in range(len(num_plot)):
    for facti in range(num_factors):
        plt.subplot(num_factors, len(num_plot), (1+obsi) + facti * len(num_plot))
        plt.imshow(observations[facti][num_plot[obsi], :, :], cmap='gray')
        plt.axis('off')

#%% Init and fit Model

# Fit Parameters
fit_params = {"ite_max": 1000}

# Init Model
model = UnstructuredRecognition(num_digits, observations, fit_params=fit_params)

# Fit model
model.fit(observations)




#%% Load and summarize

# Save Model
data_name = 'MNIST_UR_model' + datetime.now().strftime("%Y_%M_%d_%Hh%Mm%Ss") + '.pkl'
print("date and time =", data_name)
with open(data_name, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(model, f)

# Load Model
with open('MNIST_UR_model_99acc.pkl', 'rb') as f:
    model = pickle.load(f)

#%%

# Deactivate dropouts
model.recognition_network.eval()

# Grasp Test set
test_images = test_data.test_data[torch.isin(test_data.test_labels, sub_ids)]
test_labels = test_data.test_labels[torch.isin(test_data.test_labels, sub_ids)]

# Reduce training set
reduce_training_set = 20000

# Convert Test datasets
test_tmp = torch.tensor(test_images.clone().detach(), dtype=torch.float32)

# Use Recognition Network to classify digits
train_predictions =\
    torch.argmax(model.recognition_network.forward(train_images[:reduce_training_set].unsqueeze(dim=1)), dim=1)
test_predictions = \
    torch.argmax(model.recognition_network.forward(test_tmp.unsqueeze(dim=1)), dim=1)

# Find best permutation between model clusters and digits identity
perm_opt = model.permute_prediction(train_predictions, train_labels[:reduce_training_set], sub_ids)

# Permute Labels
train_predictions = perm_opt[train_predictions]
test_predictions = perm_opt[test_predictions]

# Train / Test performances
train_accuracy = sum(abs(train_predictions-train_labels[:reduce_training_set]) < 0.1) / reduce_training_set
test_accuracy = sum(abs(test_predictions-test_labels) < 0.01 ) / len(test_labels)

# Summary
train_results = str(np.round(train_accuracy.numpy(), 2))
test_results = str(np.round(test_accuracy.numpy(), 2))

# Plot And print summary
plt.figure()
plt.plot(model.loss_tot)
plt.title('Accuracy Train / Test = ' + test_results + ' / ' + test_results)

print('Training Accuracy = ' + train_results)
print('Testing  Accuracy = ' + test_results)




