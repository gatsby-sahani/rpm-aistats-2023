from unstructured_recognition_LDA import UnstructuredRecognition
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle

# Load rearranged van Hateren dataset
with open('./../RPM_data/LDA/LDA_observations.pkl', 'rb') as f:
   [observation_list, observations, img_tot] = pickle.load(f)


#%%
# Reproducibility
torch.manual_seed(1)

# Dimensions of the problem
num_obs = observation_list[0].shape[0]
dim_patch = observation_list[0].shape[-1]
num_textures = 10

# Fit Parameters
fit_params = {"ite_max": 10000}

# Init Model
model = UnstructuredRecognition(num_textures, observation_list, fit_params=fit_params)

# Fit model
model.fit(observation_list)


#Save Model
# from datetime import datetime
# data_name = 'rpm_lda' + datetime.now().strftime("%Y_%M_%d_%Hh%Mm%Ss") + '.pkl'
# print("date and time =", data_name)
# with open(data_name, 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump(model, f)

