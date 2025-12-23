#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 13:43:25 2025

@author: raluca
"""

import numpy as np
import pickle as pkl
import pandas as pd
import torch 
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
with open('final_results_without_SMC.pkl', 'rb') as file:
    data_without = pkl.load(file)
    
#golden_seed = np.load('golden_seed.npy').flatten()
    
loss_without = data_without['loss']
parameters_without = data_without['parameters']
histories_without = data_without['histories']
running_times_without = data_without['running_times']
regularization_params_without = data_without['regularization_params']



with open('final_results_with_SMC.pkl', 'rb') as file:
    data_with = pkl.load(file)
    
    
loss_with = data_with['loss']
parameters_with = data_with['parameters']
histories_with = data_with['histories']
running_times_with = data_with['running_times']
regularization_params_with = data_with['regularization_params']

labels = ["Gradient only", "ABC → Gradient"]
t_stat, running_time_p_value = ttest_ind(running_times_with, running_times_without, equal_var=False)

plt.figure(figsize=(10, 6))

bins = np.linspace(
    min(running_times_without.min(), running_times_with.min())/60,
    max(running_times_without.max(), running_times_with.max())/60,
    50
)

plt.hist(running_times_without/60, bins=bins, alpha=0.6,
         label=labels[0], color='tab:blue', edgecolor='black')

plt.hist(running_times_with/60, bins=bins, alpha=0.6,
         label=labels[1], color='tab:orange', edgecolor='black')

plt.xlabel("Running time per seed [minutes]", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title(f"Histogram of Running Times (p = {running_time_p_value:.3g})", fontsize=15)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()



bins = np.linspace(
    min(np.log10(loss_without).min(), np.log10(loss_with).min()),
    max(np.log10(loss_without).max(), np.log10(loss_with).max()),
    50
)
plt.hist(np.log10(loss_without), bins=bins, alpha=0.6,
         label=labels[0], color='tab:blue', edgecolor='black')

plt.hist(np.log10(loss_with), bins=bins, alpha=0.6,
         label=labels[1], color='tab:orange', edgecolor='black')

plt.xlabel(r"$\log_{10}(\mathrm{Loss})$", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("Histogram of Loss Values Across Seeds", fontsize=15)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


MLE_index_without = np.argmin(loss_without)
MLE_without = parameters_without[MLE_index_without,:]

MLE_index_with = np.argmin(loss_with)
MLE_with = parameters_with[MLE_index_with,:]


np.save('MLE_without_SMC', MLE_without)

np.save('MLE_with_SMC', MLE_with)




