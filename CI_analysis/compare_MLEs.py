#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 12:17:09 2025

@author: raluca
"""

import numpy as np
import matplotlib.pyplot as plt


parameter_names = [
    r"$D$", r"$\lambda_g$", 
    r"$\rho_1^{\mathrm{fib}_1}$", r"$\rho_2^{\mathrm{fib}_1}$", r"$\rho_3^{\mathrm{fib}_1}$",
    r"$\rho_1^{\mathrm{fib}_2}$", r"$\rho_2^{\mathrm{fib}_2}$", r"$\rho_3^{\mathrm{fib}_2}$",
    #r"$\rho_1^{\mathrm{fib}_3}$", r"$\rho_2^{\mathrm{fib}_3}$", r"$\rho_3^{\mathrm{fib}_3}$",
    r"$\rho_1^{\mathrm{mac}}$", r"$\rho_2^{\mathrm{mac}}$", r"$\rho_3^{\mathrm{mac}}$",
    r"$\rho_1^{\mathrm{EC}}$", r"$\rho_2^{\mathrm{EC}}$", r"$\rho_3^{\mathrm{EC}}$",
    r"$K_1$", r"$K_2$", r"$K_3$",
    r"$R_1^{\mathrm{fib}_1}$", r"$R_2^{\mathrm{fib}_1}$", r"$R_3^{\mathrm{fib}_1}$",
    r"$R_1^{\mathrm{fib}_2}$", r"$R_2^{\mathrm{fib}_2}$", r"$R_3^{\mathrm{fib}_2}$",
   # r"$R_1^{\mathrm{fib}_3}$", r"$R_2^{\mathrm{fib}_3}$", r"$R_3^{\mathrm{fib}_3}$",
    r"$R_1^{\mathrm{mac}}$", r"$R_2^{\mathrm{mac}}$", r"$R_3^{\mathrm{mac}}$",
    r"$R_1^{\mathrm{EC}}$", r"$R_2^{\mathrm{EC}}$", r"$R_3^{\mathrm{EC}}$"
]

units = [
    r"$\mathrm{m^2/s}$", r"$\mathrm{s^{-1}}$"]+\
    [r"$\mathrm{mol/(cell \cdot s)}$"] * 12 + \
    [r"$\mathrm{m^3/(s \cdot mol)}$"] * 3 + \
    [r"$\mathrm{mols/cell}$"] * 12

MLE_wound = np.load('MLE_with_SMC_wound.npy')
MLE_scalp = np.load('MLE_with_SMC_scalp.npy')

CIs_wound = np.load('wound_SMC_relative_CIs.npy')
CIs_scalp = np.load('CIs_SMC_scalp.npy')


different_indices = np.array([8,9,10,26,27,28])
extracted_indices= np.setdiff1d(np.arange(len(MLE_wound)), different_indices)
MLE_wound_extracted = MLE_wound[extracted_indices]
CIs_wound_extracted = CIs_wound[extracted_indices]

CIs_scalp*=1.96*0.04
CIs_wound_extracted*=1.96*0.04

p = len(parameter_names)

for i in range(p):
    name = parameter_names[i]

    mle_wound = MLE_wound_extracted[i]
    mle_scalp = MLE_scalp[i]

    ci_wound = CIs_wound_extracted[i]
    ci_scalp = CIs_scalp[i]

    plt.figure(figsize=(4,10))   # tall figure
    # --- X positions (closer together) ---
    x_wound = 0.0
    x_scalp = 0.4   # instead of 1.0
    
    lower_wound = np.maximum(0, mle_wound - ci_wound)
    upper_wound = mle_wound + ci_wound

# Convert these into yerr format
    yerr_wound = np.vstack([mle_wound - lower_wound, upper_wound - mle_wound])

    # --- Plot MLE + CI (wound) ---
    plt.errorbar(
        x=[x_wound],
        y=[mle_wound],
        yerr=yerr_wound,
        fmt='o',
        markersize=10,            # larger marker
        markeredgewidth=2.5,      # thicker border
        capsize=7,                # longer caps
        elinewidth=2.5,           # thicker CI lines
        linewidth=2,              # thicker marker line
        label='MLE wound'
    )
    
    lower_scalp = np.maximum(0, mle_scalp - ci_scalp)
    upper_scalp = mle_scalp + ci_scalp

# Convert these into yerr format
    yerr_scalp = np.vstack([mle_scalp - lower_scalp, upper_scalp - mle_scalp])

    # --- Plot MLE + CI (scalp) ---
    plt.errorbar(
        x=[x_scalp],
        y=[mle_scalp],
        yerr=yerr_scalp,
        fmt='s',
        markersize=10,            # larger marker
        markeredgewidth=2.5,      # thicker border
        capsize=7,                # longer caps
        elinewidth=2.5,           # thicker CI lines
        linewidth=2,              # thicker marker line
        label='MLE scalp'
    )

    # --- Formatting ---
    plt.xticks([x_wound, x_scalp], ['Wound', 'Scalp'], fontsize=14)
    plt.ylabel(f"{name} [{units[i]}]", fontsize=14)
   # plt.title(f"Parameter: {name}")
    plt.grid(alpha=0.3)
    #plt.legend()

    plt.tight_layout()
    plt.show()
    
    
    
MLE_wound = np.load('MLE_with_SMC_wound.npy')  
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
CIs_wound = np.load('wound_SMC_relative_CIs.npy')



# Parameter index mapping
beta1 = [2, 5, 8]   # FB-I, FB-II, FB-III
beta2 = [3, 6, 9]
beta3 = [4, 7, 10]

isoforms = [beta1, beta2, beta3]
labels_iso = [r"$\beta_1$", r"$\beta_2$", r"$\beta_3$"]
colors = ["blue", "red", "black"]
markers = ["o", "s", "D"]

# x positions for FB-I, FB-II, FB-III
x_base = np.array([0, 1, 2])
x_labels = ["Mesemchymal/Myofib", "Proinflammatory", "Papillary"]

plt.figure(figsize=(10,6))

for iso_idx, idx_list in enumerate(isoforms):

    iso_label = labels_iso[iso_idx]
    color = colors[iso_idx]
    marker = markers[iso_idx]

    # small shifts so all isoforms appear within the FB group
    x_shift = iso_idx * 0.25 - 0.25

    for k, param_index in enumerate(idx_list):

        mle = MLE_wound[param_index]
        ci  = CIs_wound[param_index]

        # ---- Clip CI ----
        lower = max(0, mle - ci)
        upper = mle + ci
        yerr = np.array([[mle - lower], [upper - mle]])

        plt.errorbar(
            x_base[k] + x_shift, mle, yerr=yerr,
            fmt=marker, color=color,
            markersize=10, markeredgewidth=2.5,
            elinewidth=2.5, capsize=6,
            label=iso_label if k == 0 else ""
        )

plt.xticks(x_base, x_labels, fontsize=14)
plt.ylabel(f"Production rate {units[2]}", fontsize=12)
#plt.title("TGF-β Isoform Production Across Fibroblast Subtypes", fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()

plt.show()





