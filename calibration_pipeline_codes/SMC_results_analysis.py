#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 15:44:38 2025

@author: raluca
"""

from sklearn.decomposition import PCA
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from textwrap import shorten

particle_realizations = np.load("accepted_realizations.npy")
accepted_scores = np.load("accepted_scores.npy")
#golden_seed = np.load('golden_seed.npy')
epsilons = np.load('population_epsilons.npy')

plt.plot(range(len(epsilons)), epsilons, marker='o')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Value against Index')
plt.grid(True)
plt.show()


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

with open("prior_saved_data.pkl", "rb") as f:
    loaded_data = pkl.load(f)

fixed_param_indices = loaded_data["fixed_parameter_indices"]
mean_prior = loaded_data["variable_parm_means"]
std_prior = loaded_data["variable_parm_stds"]
#gmm_prior = loaded_data["prior_gmm"]
#fixed_param_MLE = loaded_data["fixed_variable_MLE"]    

var_param_num = len(mean_prior)
population_number = particle_realizations.shape[0]

d= len(mean_prior)#len(fixed_param_MLE) + len(mean_prior)
#var_param_indices = np.setdiff1d(np.arange(d),fixed_param_indices)
#parameter_names_var = [parameter_names[i] for i in var_param_indices]
#units_var = [units[i] for i in var_param_indices]

mapped_var_realizations = np.zeros_like(particle_realizations)
for i in range(var_param_num):
    mapped_var_realizations[:,:,i] = np.exp(particle_realizations[:,:,i]*std_prior[i] +mean_prior[i])
    
for i in range(var_param_num):
   # true_value = golden_seed.flatten()[i]
    fig, axes = plt.subplots(5, 5, figsize=(16, 9))
    axes = axes.flatten()
    axes_count = 0

    # Determine common x-limits for all subplots
    x_min, x_max = np.inf, -np.inf
    for j in range(population_number):
        if (j % 5 == 0 or j == population_number - 1):  # Same condition in both loops
            data = mapped_var_realizations[j, :, i]
            x_min = min(x_min, data.min())
            x_max = max(x_max, data.max())

    for j in range(population_number):
        if (j % 5 == 0 or j == population_number - 1):
            if axes_count >= len(axes):  # FIXED CONDITION
                break

            data = mapped_var_realizations[j, :, i]
            axes[axes_count].hist(data, bins=50, density=True, alpha=0.6)
          #  axes[axes_count].axvline(true_value, color='red', linestyle='--', linewidth=2)
            axes[axes_count].set_xlim(x_min, x_max)
            axes[axes_count].set_ylabel("PDF")
            axes[axes_count].set_xlabel("Parameter Values")
            axes[axes_count].set_title(f"population number = {j}")
            axes_count += 1

    # Hide unused axes if any
    for k in range(axes_count, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    plt.suptitle(f'Histograms for Parameter: {parameter_names[i]} ({units[i]})', fontsize=14, y=1.02)
    plt.show()







net_scores_pos = accepted_scores[-1,:]
idx_desc = np.argsort(net_scores_pos)[-1000:][::-1]
final_realizations = mapped_var_realizations[-1,idx_desc,:]
np.save("starting_seeds.npy", final_realizations)
net_realizations_norm = particle_realizations[-1,:,:]

    
    
# Step 1: Fit PCA
pca = PCA()
pca.fit(net_realizations_norm)

# Step 2: Transform into PC space
posterior_pc_space = pca.transform(net_realizations_norm)  # shape: N x d

# Step 3: Get eigenvectors (principal directions)
eigenvectors = pca.components_  # shape: d x d, ordered by importance; highest variance first


# Step 4: Explained variance
explained_variance = pca.explained_variance_ratio_  # shape: d
explained_variance_cum = np.array([np.sum(explained_variance[:i+1]) for i in range(len(explained_variance))])
# (Optional) Cumulative explained variance

least_variance_PCs = eigenvectors[-7:,:]
top_indices = []
for i, elem in enumerate(least_variance_PCs):
    top_indices.append(np.argsort(elem[i])[-10:][::-1])
# least_variance_PC = eigenvectors[-1]
# second_least_variance_PC = eigenvectors[-2]
# third_least_variance_PC = eigenvectors[-3]
# fourth_least_variance_PC = eigenvectors[-3]
# fifth_least_variance_PC = eigenvectors[-3]
# sixth_least_variance_PC = eigenvectors[-3]
# seventh_least_variance_PC = eigenvectors[-3]

# selected_indices = []
# for i, elem in enumerate(top_indices):
#     denom = 

highest_variance_PC = eigenvectors[1]
second_highest_variance_PC = eigenvectors[2]


top10_indices_high_1 = np.argsort(highest_variance_PC)[-10:][::-1]
top10_indices_high_2 = np.argsort(second_highest_variance_PC)[-10:][::-1]

# Get the corresponding top 10 values
highest_var_comp_1 = highest_variance_PC[top10_indices_high_1]
highest_var_comp_2 = second_highest_variance_PC[top10_indices_high_2]


least_variance_PC_importance = np.abs(least_variance_PC)
least_variance_PC_importance /=np.sum(least_variance_PC_importance)



highest_variance_PC_importance = np.abs(highest_variance_PC)
highest_variance_PC_importance /=np.sum(highest_variance_PC_importance)

second_highest_variance_PC_importance = np.abs(second_highest_variance_PC)
second_highest_variance_PC_importance /=np.sum(second_highest_variance_PC_importance)

a1 = np.where(least_variance_PC_importance>0.03)[0]

#compute variance_reduction
T0_SMC_distribution = particle_realizations[0,:,:]
T_end_SMC_distribution = particle_realizations[-1,:,:]
T0_var = np.var(T0_SMC_distribution, axis=0)
T_end_var = np.var(T_end_SMC_distribution, axis=0)
var_reduction = (T0_var-T_end_var)/T0_var    
var_reduction_parameter_indices = np.where(var_reduction>0.65)[0]




golden_realizations = particle_realizations[-1,np.argsort(net_scores_pos)[-int(1e3):][::-1],:]
golden_realizations= golden_realizations*std_prior+mean_prior
golden_full_realizations = np.zeros([golden_realizations.shape[0],d])
#golden_full_realizations[:,fixed_param_indices] = fixed_param_MLE
#golden_full_realizations[:,var_param_indices] = golden_realizations
binary_indices_full = np.zeros(d)
#binary_indices_var = np.zeros(len(var_param_indices))
final_var_indices_local = np.intersect1d(var_reduction_parameter_indices,a)
binary_indices_var[final_var_indices_local]= 1
binary_indices_full[var_param_indices] = binary_indices_var

np.savez("golden_realization", realization = golden_full_realizations, binary = binary_indices_full)

un_norm_realizations = particle_realizations[-1,:,:]*std_prior+mean_prior
full_realizations = np.zeros((particle_realizations.shape[1],d))
full_realizations[:,fixed_param_indices] = fixed_param_MLE
full_realizations[:,var_param_indices] = un_norm_realizations
full_realizations = full_realizations[np.argsort(net_scores_pos)[-100:][::-1],:]
np.save("realizations_for_ML.npy",full_realizations)




#plot PCA results


# Data
labels = parameter_names_var
sizes_1 = least_variance_PC_importance
sizes_2 = second_least_variance_PC_importance

sorted_indices_1 = np.argsort(sizes_1)[::-1]
sorted_sizes_1 = [sizes_1[i] for i in sorted_indices_1]
sorted_labels_1 = [labels[i] for i in sorted_indices_1]

sorted_indices_2 = np.argsort(sizes_2)[::-1]
sorted_sizes_2 = [sizes_2[i] for i in sorted_indices_2]
sorted_labels_2 = [labels[i] for i in sorted_indices_2]

top_n = 5
show_pct = [True if i < top_n else False for i in range(len(sorted_sizes_1))]

# Define custom autopct function
def make_autopct(show_flags):
    def autopct(pct):
        i = autopct.counter
        autopct.counter += 1
        return f"{pct:.1f}%" if show_flags[i] else ''
    autopct.counter = 0
    return autopct


fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(
    sorted_sizes_1,
    #colors=sorted_colors,
    autopct=make_autopct(show_pct),
    startangle=90
)

ax.axis('equal')  # keep pie circular

# Legend in order of size
ax.legend(
    wedges[:top_n],                # Only top 5 wedges
    sorted_labels_1[:top_n],         # Only top 5 labels
    title="Largest components of last PCA Eigenvector",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1)
)
plt.show()



fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(
    sorted_sizes_2,
    #colors=sorted_colors,
    autopct=make_autopct(show_pct),
    startangle=90
)
ax.axis('equal')  # keep pie circular
# Legend in order of size
ax.legend(
    wedges[:top_n],                # Only top 5 wedges
    sorted_labels_2[:top_n],         # Only top 5 labels
    title="Largest components of second to last PCA Eigenvector",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1)
)
plt.show()

categories = np.array(parameter_names_var)[var_reduction>0]
values = var_reduction[var_reduction>0]*100
plt.bar(categories,values)
# Add labels and title
plt.xticks(rotation=90)
plt.xlabel('Parameters')
plt.ylabel('Percentage Reduction in Variance')
# Show the plot
plt.show()
