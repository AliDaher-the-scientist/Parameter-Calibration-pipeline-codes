#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 19:56:12 2025

@author: raluca
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


with open('accepted_result_2s.pkl', 'rb') as file:
    data = pkl.load(file)
    
MC_realizations = np.array(data['realizations'] )    
mean_prior = np.mean(MC_realizations,axis=0)
std_prior = np.std(MC_realizations, axis=0)

golden_seed = np.load('golden_seed.npy').flatten()
after_smc = np.load('CI_after_SMC.npz')  # has keys: 'MLE_after', 'relative_CIs_after'
baseline = np.load('CI.npz')              # has keys: 'MLE', 'relative_CIs'


# Extract arrays
MLE_after = after_smc['MLE']           # shape: (num_params,)
CIs_after = after_smc['relative_CIs']
MLE_base = baseline['MLE']
CIs_base = baseline['relative_CIs']


golden_seed_scaled = (golden_seed-mean_prior)/std_prior
# golden_dlambda = golden_seed_scaled[0:2]
# golden_rest = golden_seed_scaled[2:]
# golden_rest_reshaped = golden_rest.reshape(-1, 3).sum(axis=1)
# reshaped_golden_seed = np.concatenate([golden_dlambda, golden_rest_reshaped])

MLE_after_scaled = (np.load('MLE_after_SMC.npy')-mean_prior)/std_prior
MLE_before_scaled = (np.load('MLE.npy')-mean_prior)/std_prior

num_params = len(MLE_after_scaled)
indices = np.array([ 0,  1,  4,  7, 19, 22, 25, 26, 28, 31, 34])
labels = [f'Param {i+1}' for i in indices]  # or use your parameter names

x = indices
width = 0.27

plt.figure(figsize=(14,5))
plt.bar(x - width, MLE_before_scaled[indices], width, label='MLE Before SMC')
plt.bar(x, MLE_after_scaled[indices], width, label='MLE After SMC')
plt.bar(x + width, golden_seed_scaled[indices], width, label='True Value (golden seed)')

plt.xlabel('Parameter')
plt.ylabel('Scaled Value')
plt.title('MLE and True Value (scaled parameters)')
plt.xticks(x, labels, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

plt.plot(x, MLE_before_scaled[indices], 'o-', label='MLE Before SMC')
plt.plot(x, MLE_after_scaled[indices], 's-', label='MLE After SMC')
plt.plot(x, golden_seed_scaled[indices], '*-', label='True Value')
plt.legend()
plt.show()

diff_SMC = np.abs(MLE_after_scaled-golden_seed_scaled)/golden_seed_scaled
diff_before = np.abs(MLE_before_scaled-golden_seed_scaled)/golden_seed_scaled



# num_params = len(MLE_after_scaled)-1
# labels = [f'Param {i+1}' for i in range(num_params)]  # or use your parameter names
# x = np.arange(num_params)
# width = 0.27
# MLE_after_scaled_diff = (MLE_after_scaled-golden_seed_scaled)/golden_seed_scaled
# MLE_before_scaled_diff = (MLE_before_scaled-golden_seed_scaled)/golden_seed_scaled

# a = np.delete(MLE_after_scaled_diff,28)
# b = np.delete(MLE_before_scaled_diff,28)

# plt.figure(figsize=(14,5))
# plt.bar(x - width, b, width, label='MLE Before SMC')
# plt.bar(x, a, width, label='MLE After SMC')


# plt.xlabel('Parameter')
# plt.ylabel('Relative Difference')
# #plt.title('MLE and True Value (scaled parameters)')
# plt.xticks(x, labels, rotation=45)
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.plot(x, b, 'o-', label='MLE Before SMC')
# plt.plot(x, a, 's-', label='MLE After SMC')
# plt.legend()
# plt.show()



# num_params = len(MLE_after)
# x = np.arange(num_params-2)
# plt.figure(figsize=(14,5))

# # Plot baseline CIs
plt.errorbar(x-0.15, MLE_before_scaled[indices], yerr=CIs_base[indices], fmt='o', label='MLE Gradient', capsize=2)
# # Plot after SMC CIs
plt.errorbar(x+0.15, MLE_after_scaled[indices], yerr=CIs_after[indices], fmt='o', label='MLE SMC then Gradient', capsize=2)
# # Plot golden_seed
plt.scatter(x, golden_seed_scaled[indices], marker='*', s=100, c='gold', label='True Value (golden_seed)')

plt.xlabel('Parameter Index')
plt.ylabel('Parameter Value')
plt.title('MLE and Relative CI for Each Parameter')
plt.legend()
plt.tight_layout()
plt.show()

identifiable =   np.array([ 0,  1,  4,  7, 10, 13, 16, 17, 18, 19, 20, 22, 23, 25, 26, 28, 29,
       31, 32, 34]) #15 threshold

non_identifiable = np.array([ 2,  3,  5,  6,  8,  9, 11, 12, 14, 15, 21, 24, 27, 30, 33])

better_MLE_indices =  np.where(diff_SMC<=diff_before+0.1)[0]
better_CI_indices = np.where(CIs_after<=CIs_base+0.1)[0]
intersect_indices = np.intersect1d(better_MLE_indices, better_CI_indices)

group_A = np.where(CIs_base<10)[0]
group_B = np.where(CIs_base>10)[0]


group_A_intersect = identifiable#np.intersect1d(identifiable,better_CI_indices)#np.intersect1d(intersect_indices, group_A)
group_B_intersect = non_identifiable#np.intersect1d(non_identifiable, better_CI_indices) #np.intersect1d(intersect_indices, group_B)
#group_C_intersect = np.intersect1d(intersect_indices, group_C)

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


offset = 0.2
x = np.arange(len(group_A_intersect))
plt.errorbar(x-0.2, MLE_before_scaled[group_A_intersect], yerr=CIs_base[group_A_intersect], fmt='o', label='MLE Gradient', c='black', capsize=2)
plt.errorbar(x+0.2, MLE_after_scaled[group_A_intersect], yerr=CIs_after[group_A_intersect], fmt='o', label='MLE SMC then Gradient', c='blue',capsize=2)
plt.scatter(x, golden_seed_scaled[group_A_intersect], marker='*', s=150, c='red', label='True Value (golden_seed)')

plt.xlabel('Parameter')
plt.ylabel('Z-transformed Parameter Value')
plt.title('MLE and Relative CI for Each Parameter')

# Set x-axis ticks and labels
plt.xticks(x, [parameter_names[i] for i in group_A_intersect], rotation=45, ha='right')

plt.legend()
plt.tight_layout()
plt.show()



x = np.arange(len(group_B_intersect))
plt.errorbar(x-0.2, MLE_before_scaled[group_B_intersect], yerr=CIs_base[group_B_intersect], fmt='o', label='MLE Gradient', c='black',capsize=2)
plt.errorbar(x+0.2, MLE_after_scaled[group_B_intersect], yerr=CIs_after[group_B_intersect], fmt='o', label='MLE SMC then Gradient',c='blue', capsize=2)
plt.scatter(x, golden_seed_scaled[group_B_intersect], marker='*', s=150, c='red', label='True Value (golden_seed)')

plt.xlabel('Parameter')
plt.ylabel('Z-transformed Parameter Value')
plt.title('MLE and Relative CI for Each Parameter')

# Set x-axis ticks and labels
plt.xticks(x, [parameter_names[i] for i in group_B_intersect], rotation=45, ha='right')

plt.legend()
plt.tight_layout()
plt.show()


