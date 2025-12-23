#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 15:56:40 2025

@author: raluca
"""

import numpy as np 
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.stats import ks_2samp
with open('accepted_results.pkl', 'rb') as file:
    data = pkl.load(file)
    
accepted_scores = np.array(data["scores"])
accepted_realizations = np.array(data["realizations"])  
    
bin_width = 0.05
bins = np.arange(min(accepted_scores), max(accepted_scores) + bin_width, bin_width)    

plt.hist(accepted_scores, bins=bins, edgecolor='black')
plt.title('Accepted Scores histogram MCF run')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Frequency')
plt.show()



accepted_realizations = np.array(accepted_realizations) 
accepted_scores = np.array(accepted_scores)
filtered_indices = np.argsort(accepted_scores)[-1000:]
filtered_realizations = accepted_realizations[filtered_indices,:]
filtered_scores = accepted_scores[filtered_indices]


MC = int(9e5)
# Sample from prior
rec_const = 6.0214*10**23 #avocadro's constant

# Sample from prior
D = 10**(np.random.uniform(np.log10(1.0)-11, np.log10(2)-10, MC)) #m^2/s
lambda_g = np.random.uniform(2e-6, 2e-5, MC) #1/s


K1_2 = 10**(np.random.uniform(np.log10(2)+2, np.log10(5)+4, MC))
K2_3 = 10**(np.random.uniform(np.log10(5)+2, np.log10(2)+4, MC)) #in m^3/(s.mol)
K3_2 = 10**(np.random.uniform(np.log10(5)+2, np.log10(10)+4, MC))

rho_fib1 = np.random.uniform(1.0,4,MC)*1e-22
partition_fib1 = np.random.dirichlet([1, 1, 1], size=MC)
rho_fib1_1 = partition_fib1[:,0]* rho_fib1
rho_fib1_2 = partition_fib1[:,1]* rho_fib1
rho_fib1_3 = partition_fib1[:,2]* rho_fib1

rho_fib2 = np.random.uniform(1.0,4,MC)*1e-22
partition_fib2 = np.random.dirichlet([1, 1, 1], size=MC)
rho_fib2_1 = partition_fib2[:,0]* rho_fib2
rho_fib2_2 = partition_fib2[:,1]* rho_fib2
rho_fib2_3 = partition_fib2[:,2]* rho_fib2

rho_fib3 = np.random.uniform(1.0,4.0,MC)*1e-22
partition_fib3 = np.random.dirichlet([1, 1, 1], size=MC)
rho_fib3_1 = partition_fib3[:,0]* rho_fib3
rho_fib3_2 = partition_fib3[:,1]* rho_fib3
rho_fib3_3 = partition_fib3[:,2]* rho_fib3

rho_mac = 0.5*np.random.uniform(1.0,4.0,MC)*1e-22
partition_mac = np.random.dirichlet([1, 1, 1], size=MC)
rho_mac_1 = partition_mac[:,0]* rho_mac
rho_mac_2 = partition_mac[:,1]* rho_mac
rho_mac_3 = partition_mac[:,2]* rho_mac

rho_endo = 0.2*np.random.uniform(1.0,4.0,MC)*1e-22
partition_endo = np.random.dirichlet([1, 1, 1], size=MC)
rho_endo_1 = partition_endo[:,0]* rho_endo
rho_endo_2 = partition_endo[:,1]* rho_endo
rho_endo_3 = partition_endo[:,2]* rho_endo

#according to ligand numbering not receptor numbering
rec_number_fib1 = np.random.uniform(8000,15000,MC)/rec_const
partition_rec_fib1 = np.random.dirichlet([1, 1, 1], size=MC)
rec_number_fib1_1 = partition_rec_fib1[:,0]* rec_number_fib1
rec_number_fib1_2 = partition_rec_fib1[:,1]* rec_number_fib1
rec_number_fib1_3 = partition_rec_fib1[:,2]* rec_number_fib1

rec_number_fib2 = np.random.uniform(8000,15000,MC)/rec_const
partition_rec_fib2 = np.random.dirichlet([1, 1, 1], size=MC)
rec_number_fib2_1 = partition_rec_fib2[:,0]* rec_number_fib2
rec_number_fib2_2 = partition_rec_fib2[:,1]* rec_number_fib2
rec_number_fib2_3 = partition_rec_fib2[:,2]* rec_number_fib2

rec_number_fib3 = np.random.uniform(8000,15000,MC)/rec_const
partition_rec_fib3 = np.random.dirichlet([1, 1, 1], size=MC)
rec_number_fib3_1 = partition_rec_fib3[:,0]* rec_number_fib3
rec_number_fib3_2 = partition_rec_fib3[:,1]* rec_number_fib3
rec_number_fib3_3 = partition_rec_fib3[:,2]* rec_number_fib3

rec_number_ECs = np.random.uniform(6000,14000,MC)/rec_const
partition_rec_ECs = np.random.dirichlet([1, 1, 1], size=MC)
rec_number_ECs_1 = partition_rec_ECs[:,0]* rec_number_ECs
rec_number_ECs_2 = partition_rec_ECs[:,1]* rec_number_ECs
rec_number_ECs_3 = partition_rec_ECs[:,2]* rec_number_ECs

rec_number_mac = (10**(np.random.uniform(np.log10(3.0)+2, np.log10(10)+3, MC)))/rec_const
partition_rec_mac = np.random.dirichlet([1, 1, 1], size=MC)
rec_number_mac_1 = partition_rec_mac[:,0]* rec_number_mac
rec_number_mac_2 = partition_rec_mac[:,1]* rec_number_mac
rec_number_mac_3 = partition_rec_mac[:,2]* rec_number_mac





old_realizations = np.column_stack([D, lambda_g, rho_fib1_1, rho_fib1_2, rho_fib1_3, \
                            rho_fib2_1, rho_fib2_2, rho_fib2_3, \
                            rho_fib3_1, rho_fib3_2, rho_fib3_3,\
                            rho_mac_1, rho_mac_2, rho_mac_3, \
                            rho_endo_1, rho_endo_2, rho_endo_3, \
                            K1_2, K2_3, K3_2,\
                            rec_number_fib1_1, rec_number_fib1_2, rec_number_fib1_3,\
                            rec_number_fib2_1, rec_number_fib2_2, rec_number_fib2_3,\
                            rec_number_fib3_1, rec_number_fib3_2, rec_number_fib3_3,\
                            rec_number_mac_1, rec_number_mac_2, rec_number_mac_3,\
                            rec_number_ECs_1, rec_number_ECs_2, rec_number_ECs_3])
 
parameter_names = [
    r"$D$", r"$\lambda_g$", 
    r"$\rho_1^{\mathrm{fib}_1}$", r"$\rho_2^{\mathrm{fib}_1}$", r"$\rho_3^{\mathrm{fib}_1}$",
    r"$\rho_1^{\mathrm{fib}_2}$", r"$\rho_2^{\mathrm{fib}_2}$", r"$\rho_3^{\mathrm{fib}_2}$",
    r"$\rho_1^{\mathrm{fib}_3}$", r"$\rho_2^{\mathrm{fib}_3}$", r"$\rho_3^{\mathrm{fib}_3}$",
    r"$\rho_1^{\mathrm{mac}}$", r"$\rho_2^{\mathrm{mac}}$", r"$\rho_3^{\mathrm{mac}}$",
    r"$\rho_1^{\mathrm{EC}}$", r"$\rho_2^{\mathrm{EC}}$", r"$\rho_3^{\mathrm{EC}}$",
    r"$K_1$", r"$K_2$", r"$K_3$",
    r"$R_1^{\mathrm{fib}_1}$", r"$R_2^{\mathrm{fib}_1}$", r"$R_3^{\mathrm{fib}_1}$",
    r"$R_1^{\mathrm{fib}_2}$", r"$R_2^{\mathrm{fib}_2}$", r"$R_3^{\mathrm{fib}_2}$",
    r"$R_1^{\mathrm{fib}_3}$", r"$R_2^{\mathrm{fib}_3}$", r"$R_3^{\mathrm{fib}_3}$",
    r"$R_1^{\mathrm{mac}}$", r"$R_2^{\mathrm{mac}}$", r"$R_3^{\mathrm{mac}}$",
    r"$R_1^{\mathrm{EC}}$", r"$R_2^{\mathrm{EC}}$", r"$R_3^{\mathrm{EC}}$"
]

# LaTeX-formatted units
units = [
    r"$\mathrm{m^2/s}$", r"$\mathrm{s^{-1}}$"]+\
    [r"$\mathrm{mol/(cell \cdot s)}$"] * 15 + \
    [r"$\mathrm{m^3/(s \cdot mol)}$"] * 3 + \
    [r"$\mathrm{mols/cell}$"] * 15

# Sanity check
assert len(parameter_names) == len(units)

p_values = []
# Loop over parameters
d = old_realizations.shape[1]
for i in range(d):
    old_realization = old_realizations[:,i]
    new_realization = filtered_realizations[:,i]    
    statistic, p_value = ks_2samp(old_realization, new_realization)
    p_values.append(p_value)
    # plt.figure(figsize=(6, 4))
    # plt.hist(old_realization, bins=50, alpha=0.6, label="Original", color="gray", density=True)
    # plt.hist(new_realization, bins=50, alpha=0.6, label="Filtered", color="blue", density=True)
    
    # plt.xlabel(f'{parameter_names[i]} ({units[i]})', fontsize=12)
    # plt.ylabel("Density", fontsize=12)
    # plt.title(f"p-value = {p_value:.4g}", fontsize=14)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
p_values = np.array(p_values) 
print(len(p_values[p_values>0.01]))   
parameter_names_fixed = [parameter_names[i] for i in np.where(p_values>0.01)[0]]

#variable_filtered_realizations = filtered_realizations[:, np.where(p_values<0.01)[0]]
log_var_filt_real = np.log(filtered_realizations)
mean_variable_realizations = np.mean(log_var_filt_real, axis= 0)
std_variable_realizations = np.std(log_var_filt_real, axis = 0)
normalized_variable_filtered_realizations = (log_var_filt_real-\
                                             mean_variable_realizations)/std_variable_realizations




def mode_from_hist(data, bins=50):
    """
    Estimate PDF via histogram binning and return the mode (bin center with highest density).

    Parameters:
    - data: 1D numpy array of realizations for a parameter
    - bins: number of bins for histogram

    Returns:
    - mode_value: value at the center of the bin with highest count (mode)
    """
    counts, bin_edges = np.histogram(data, bins=bins, density=True)  # density=True normalizes to PDF
    max_bin_index = np.argmax(counts)
    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mode_value = bin_centers[max_bin_index]
    return mode_value

#mode_fixed_variables = []
fixed_param_indices = np.where(p_values>=0.01)[0]
# for i,idx in enumerate(fixed_param_indices):
#     mode_fixed_variables.append(mode_from_hist(filtered_realizations[:,idx]))


    
    



prior_distribution_info = {
    "variable_parm_means": mean_variable_realizations,
    "variable_parm_stds": std_variable_realizations,
    #"fixed_variable_MLE": mode_fixed_variables,
    "fixed_parameter_indices": fixed_param_indices,
    "initial_particles": normalized_variable_filtered_realizations,
    "initial_scores": filtered_scores

}

with open("prior_saved_data_reduced.pkl", "wb") as f:
    pkl.dump(prior_distribution_info, f)
