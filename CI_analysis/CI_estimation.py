#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 12:09:58 2025

@author: raluca
"""

from mpi4py import MPI
from scipy.sparse.linalg import splu
#from scipy.sparse import csc_matrix
import torch
import numpy as np
import torch.nn as nn
#import torch.optim as optim
device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle as pkl
import pandas as pd
import time
from scipy.sparse import csc_matrix
from scipy.optimize import minimize, BFGS
from torch.autograd.functional import jacobian
import matplotlib.pyplot as plt



        
def build_system(theta, cell_densities, mesh_prop, neigbourhood_list):
        
        if type(cell_densities) ==torch.Tensor:
            np_cell_densities = cell_densities.cpu().numpy()
        else:
            np_cell_densities = cell_densities
        n= cell_densities.shape[1]
        Area, L_c, L_e = mesh_prop
        D, lambda_g = theta[[0,1]]
        param_B = theta[2:]
        rho = param_B[:n]
        K = param_B[n]
        rec_number = param_B[-n:]

        CV_num = cell_densities.shape[0]
        area, L_c, L_e = mesh_prop
        L_ratio = L_e/L_c
        A = np.zeros((CV_num, CV_num))
        b = np.zeros((CV_num))
        d = len(theta)
        dA_dtheta = np.zeros((CV_num,CV_num,d))
        db_dtheta = np.zeros((CV_num, d))
        dg_dtheta_scaled = np.zeros((d, CV_num))
        
        # A = np.zeros((CV_num, CV_num), dtype=np.complex128)
        # b = np.zeros((CV_num,), dtype=np.complex128)
        # dA_dtheta = np.zeros((CV_num, CV_num, d), dtype=np.complex128)
        # db_dtheta = np.zeros((CV_num, d), dtype=np.complex128)
        # dg_dtheta_scaled = np.zeros((d, CV_num), dtype=np.complex128)
        for i in range(CV_num):
            neighbours = neigbourhood_list[i]
            A[i,i] = D*L_ratio*len(neighbours)+area*(lambda_g +\
                     np_cell_densities[i,:]@(rec_number*K))
            b[i] = area * np_cell_densities[i,:]@rho
            A[i, neighbours] = -D*L_ratio
            db_dtheta[i,2:n+2] = area * np_cell_densities[i,:]
            dA_dtheta[i, neighbours,0] = -L_ratio
            dA_dtheta[i,i,0] = L_ratio*len(neighbours)
            dA_dtheta[i,i,1] = area
            dA_dtheta[i,i,n+2] = area*np_cell_densities[i,:]@rec_number
            dA_dtheta[i,i,-n:] = area*np_cell_densities[i,:]*K
        #scale_A = np.linalg.norm(A)
       # scale_b = np.linalg.norm()
        scale_A = 1.0 / np.linalg.norm(A, ord=2) if np.linalg.norm(A) > 0 else 1.0
        scale_b = 1.0 / np.linalg.norm(b) if np.linalg.norm(b) > 0 else 1.0
        A_scaled = csc_matrix(A)*scale_A
        b_scaled = b*scale_b
        dA_dtheta *= scale_A
        db_dtheta *= scale_b
        lu = splu(A_scaled)
        g_scaled = lu.solve(b_scaled) #in pmol/m^2
        #g=g/1e3
        for i in range(d):
            RHS = db_dtheta[:,i]- dA_dtheta[:,:,i]@g_scaled
            dg_dtheta_scaled[i,:] = lu.solve(RHS)
        g = g_scaled*scale_A/scale_b
        dg_dtheta = dg_dtheta_scaled*scale_A/scale_b
        g*=1e8
        dg_dtheta*=1e8
        return g, dg_dtheta            
    


class SolveG(torch.autograd.Function):
    
    

    @staticmethod
    def forward(ctx, theta, cell_densities, mesh_prop,\
                     neigbourhood_list):
        # Build A, b from theta
        # Also construct dA, db while looping
        theta_np = theta.detach().cpu().numpy()
        g_np , dg_dtheta_np = build_system(theta_np,\
                              cell_densities, mesh_prop, neigbourhood_list)
        g = torch.from_numpy(g_np).to(dtype=theta.dtype, device=device)
        dg_dtheta = torch.from_numpy(dg_dtheta_np).to(dtype=theta.dtype, device=device)
        ctx.save_for_backward(dg_dtheta)
        return g
    @staticmethod
    def backward(ctx, grad_output):
        dg_dtheta, = ctx.saved_tensors  # shape (N_var, M)
        grad_output = grad_output.view(-1)  # shape: (M,)
        grad_theta_full = torch.matmul(dg_dtheta,grad_output)
        # Return gradient for theta_full, None for other inputs
       # mean_grad_theta_full = torch.mean(grad_theta_full)
        #print(f"grad_theta_full norm in backward autograd function: {mean_grad_theta_full*(1/mean_grad_theta_full*grad_theta_full).norm().item()}")
        return grad_theta_full, None, None, None, None
    
class single_ligand_model(nn.Module):
    def __init__(self, neighbours_list, cell_densities, mesh_prop):
        super().__init__()
        # θ is the learnable parameter
        
        self.cell_densities = torch.tensor(cell_densities, dtype=torch.float32, device=device)
        self.neighbours_list = neighbours_list
        self.CV_num = cell_densities.shape[0]
        self.mesh_prop = mesh_prop


        

    
    def calc_int_str(self,g,full_theta):
        n=self.cell_densities.shape[1]
        rho = full_theta[2:n+2]
        K = full_theta[n+2]
        rec_number = full_theta[-n:]
        
        num_1 = self.cell_densities*rho   #Size CV by n
        denom_1 = torch.sum(num_1, axis=1)  #size CV
        prod_term = num_1 / denom_1.unsqueeze(1) #size CV by n
        absorb_array = K *torch.outer(rec_number, g) #size n by CV
        int_str = 1/self.CV_num*absorb_array@prod_term #n by n
        int_str = int_str*1e4
        return int_str
    
    

    
    
class multi_ligand_cob(single_ligand_model):    
    def __init__(self, parent_istance, z_transformed_MLE, std_prior, mean_prior,\
                  I_exp,seg_shared, seg_1, seg_2, seg_3):
        super().__init__(parent_instance.neighbours_list,
                         parent_instance.cell_densities,
                         parent_instance.mesh_prop)
        

        self.I_exp = torch.tensor(I_exp, dtype=torch.float64, device=device)  
        self.scaled_z_theta = nn.Parameter(torch.tensor(z_transformed_MLE, dtype=torch.float64, device = device))
        self.mean = torch.tensor(mean_prior, dtype = torch.float64, device = device)
        self.std = torch.tensor(std_prior, dtype = torch.float64, device = device)
        self.seg_shared = torch.tensor(seg_shared,dtype=torch.long, device = device)
        self.seg_1 = torch.tensor(seg_1,dtype=torch.long, device = device)
        self.seg_2 = torch.tensor(seg_2,dtype=torch.long, device = device)
        self.seg_3 = torch.tensor(seg_3,dtype=torch.long, device = device)
        self.cell_densities = torch.tensor(cell_densities, dtype=torch.float64, device = device)

    
    def construct_thetas_full(self, z_theta):  
        theta_unscaled = self.std*z_theta+self.mean
        theta_1 = torch.cat([theta_unscaled[self.seg_shared], theta_unscaled[self.seg_1]])
        theta_2 = torch.cat([theta_unscaled[self.seg_shared], theta_unscaled[self.seg_2]])
        theta_3 = torch.cat([theta_unscaled[self.seg_shared], theta_unscaled[self.seg_3]])
        return theta_1, theta_2, theta_3
    
    def calc_residual_vector(self, I_model_1, I_model_2, I_model_3):
        X = torch.cat([I_model_1.flatten(), I_model_2.flatten(),I_model_3.flatten()])
        Y = self.I_exp.flatten()
        covariance_biased = torch.mean(X*Y) - torch.mean(X)*torch.mean(Y)
        variance_biased = torch.mean(X**2) - (torch.mean(X))**2
        #     std_model_biased = torch.std(I_model, correction = 0)
        #     std_data_biased = torch.std(I_data, correction = 0)
        s = covariance_biased/variance_biased
        b = torch.mean(Y) - s*torch.mean(X)
        residual_vec = (s*X+b-Y)
        print(f"Correlation Coefficient is {self.calc_correlation(X)}")
        return residual_vec
    
    def calc_correlation(self, I_model):
         I_data = self.I_exp.flatten()
         covariance_biased = torch.mean(I_model*I_data) - torch.mean(I_model)*torch.mean(I_data)
         std_model_biased = torch.std(I_model, correction = 0)
         std_data_biased = torch.std(I_data, correction = 0)
         correlation = covariance_biased/(std_model_biased*std_data_biased)
         return correlation
    
    # def calc_loss_func(self, I1,I2,I3):
    #     correlation = self.calc_correlation(I1,I2,I3)
    #     loss = 1-correlation**2
    #     return loss
    

    
    
    def forward_residual(self,z_thetas):
        
         theta_1, theta_2, theta_3 = self.construct_thetas_full(z_thetas)
         g_1 = SolveG.apply(theta_1,self.cell_densities, self.mesh_prop,self.neighbours_list)
         I1 = self.calc_int_str(g_1, theta_1)    
         g_2 = SolveG.apply(theta_2,self.cell_densities, self.mesh_prop,self.neighbours_list)
         I2 = self.calc_int_str(g_2, theta_2) 
         g_3 = SolveG.apply(theta_3,self.cell_densities, self.mesh_prop,self.neighbours_list)
         I3 = self.calc_int_str(g_3, theta_3) 
         residual_vec = self.calc_residual_vector(I1,I2,I3)
         return residual_vec
    #     loss_A = self.calc_loss_func(I1, I2, I3) 
    #     loss_B = 0#self.lambda_r*self.regularization(self.scaled_log_theta)
    #     loss = loss_A +loss_B
       # print(f"loss A = {loss_A}")
        #print(f"loss B = {loss_B}")
        #return loss
    
    def forward_I_model(self,z_thetas):
        
         theta_1, theta_2, theta_3 = self.construct_thetas_full(z_thetas)
         g_1 = SolveG.apply(theta_1,self.cell_densities, self.mesh_prop,self.neighbours_list)
         I1 = self.calc_int_str(g_1, theta_1)    
         g_2 = SolveG.apply(theta_2,self.cell_densities, self.mesh_prop,self.neighbours_list)
         I2 = self.calc_int_str(g_2, theta_2) 
         g_3 = SolveG.apply(theta_3,self.cell_densities, self.mesh_prop,self.neighbours_list)
         I3 = self.calc_int_str(g_3, theta_3) 
         #residual_vec = self.calc_residual_vector(I1,I2,I3)
         return torch.concatenate([I1,I2,I3])
    #     loss_A = self.calc_loss_func(I1, I2, I3) 
    #     loss_B = 0#self.lambda_r*self.regularization(self.scaled_log_theta)
    #     loss = loss_A +loss_B
       # print(f"loss A = {loss_A}")
        #print(f"loss B = {loss_B}")
        #return loss
    

      
    




def get_constant_input():


    """
    Define variables used for all MC simulations
    """

    L_c = 84.5e-6 #from smallest distance
    hotspot_area = np.sqrt(3)/2*L_c**2
    L_e = L_c/np.sqrt(3)

    mesh_properties = [hotspot_area,L_c,L_e]
    with open('dataframe.pkl', 'rb') as file:
        neighbours_info = pkl.load(file)
        neighbours=neighbours_info.iloc[:, 1].tolist()

    q05_cell_numbers = pd.read_csv('q05_cell_abundances.csv')
    q05_cell_numbers.columns.values[0] = 'Nucleotide_ID'
    are_identical = neighbours_info['ID'].equals(q05_cell_numbers['Nucleotide_ID'])
    print("Are the columns identical?", are_identical, flush=True)


    cells_of_interest = ('FB-I', 'FB-II', 'FB-III', 'Mono-Mac', 'VE')

    # Create new tuple with prefixes
    prefix = 'q05cell_abundance_w_sf_'
    column_names = tuple(f"{prefix}{cell}" for cell in cells_of_interest)

    # Suppose df is your original DataFrame
    # Extract those columns
    cell_numbers_of_interest = q05_cell_numbers[list(column_names)].copy()

    # Rename columns to the original short names
    cell_numbers_of_interest.columns = cells_of_interest
    cell_densities = cell_numbers_of_interest.to_numpy()/hotspot_area

    ###Prepare the interaction strength data as 3D array, rows are receiving cells,
    #columns are ligand producing cells, and 3rd dimension is for the different ligands
    file_path = 'donor 3 day 30 filtered tgfb data.ods'

    # Read each sheet into a dictionary of DataFrames
    sheets = ['TGFB1-TGFBR2', 'TGFB2-TGFBR3', 'TGFB3-TGFBR2']
    dfs = {sheet: pd.read_excel(file_path, sheet_name=sheet, engine='odf') for sheet in sheets}

    # Now, we can extract the 5x5 tables from each sheet into a 3D NumPy array
    # We slice each DataFrame to remove the first row and first column

    interaction_data = np.array([dfs[sheet].iloc[:, 1:].values for sheet in sheets])


    
    
    
    return mesh_properties, neighbours, cell_densities, interaction_data


mesh_properties, neighbours_info, cell_densities, interaction_data = \
    get_constant_input()
    
with open('accepted_results.pkl', 'rb') as file:
    data = pkl.load(file)
    
MC_realizations = np.array(data['realizations'] )    
mean_prior = np.mean(MC_realizations,axis=0)
std_prior = np.std(MC_realizations, axis=0)
    




MLE_parameter_vec_without =np.load('MLE_without_SMC.npy').flatten()
MLE_parameter_vec_with =np.load('MLE_with_SMC.npy').flatten()#np.load('MLE.npy') #size d,  np.load('golden_seed.npy').flatten()
MLE_z_transformed_without = (MLE_parameter_vec_without- mean_prior)/std_prior
MLE_z_transformed_with = (MLE_parameter_vec_with- mean_prior)/std_prior
seg_shared = np.array([0,1])
seg_1 = np.array([2,5,8,11,14,17,20,23,26,29,32])
seg_2 = np.array([3,6,9,12,15,18,21,24,27,30,33])
seg_3 = np.array([4,7,10,13,16,19,22,25,28,31,34])



parent_instance = single_ligand_model(neighbours_info, cell_densities, mesh_properties)
MLE_instance_without = multi_ligand_cob(parent_instance, MLE_z_transformed_without, std_prior, mean_prior, interaction_data,\
                                seg_shared, seg_1, seg_2, seg_3)
    
MLE_z_tensor_without = torch.tensor(
    MLE_z_transformed_without,
    dtype=torch.float64,
    device=device,
    requires_grad=True
)
    
def Jacobian_wrapper_function_without(theta): #theta needs to be a tensor 
    residual_vector = MLE_instance_without.forward_residual(theta) 
    return residual_vector

residuals_without= MLE_instance_without.forward_residual(MLE_z_tensor_without).detach().cpu().numpy()
J_without = jacobian(Jacobian_wrapper_function_without, MLE_z_tensor_without)
J_without = J_without.detach().cpu().numpy()
relative_fischer_without = J_without.T@J_without
relative_cov_without = np.linalg.pinv(relative_fischer_without)
relative_CIs_without = np.sqrt(np.diag(relative_cov_without))
    
MLE_instance_with = multi_ligand_cob(parent_instance, MLE_z_transformed_with, std_prior, mean_prior, interaction_data,\
                                seg_shared, seg_1, seg_2, seg_3)
    
MLE_z_tensor_with = torch.tensor(
    MLE_z_transformed_with,
    dtype=torch.float64,
    device=device,
    requires_grad=True
)
    
def Jacobian_wrapper_function_with(theta): #theta needs to be a tensor 
    residual_vector = MLE_instance_with.forward_residual(theta) 
    return residual_vector

residuals_with= MLE_instance_with.forward_residual(MLE_z_tensor_with).detach().cpu().numpy()
J_with = jacobian(Jacobian_wrapper_function_with, MLE_z_tensor_with)
J_with = J_with.detach().cpu().numpy()
relative_fischer_with = J_with.T@J_with
relative_cov_with = np.linalg.pinv(relative_fischer_with)
relative_CIs_with = np.sqrt(np.diag(relative_cov_with))


lim = np.maximum(np.max(relative_CIs_without), np.max(relative_CIs_with))
x_linear = np.arange(0,lim,0.01)
plt.scatter(relative_CIs_without, relative_CIs_with)
plt.plot(x_linear,x_linear, 'r--')
plt.xlabel('Gradient Method Only')
plt.ylabel('ABC then Gradient Method')
plt.title('Relative Confidence Intervals of Z-Transformed Parameters')
plt.grid(True)
plt.xlim(0,lim)
plt.ylim(0,lim)
plt.show()


# MLE_gradient_only
# MLE_gradient_and_ABC
# CI_gradient_only
# CI_gradient_and_ABC
# true_parameters

MLE_gradient_only = (MLE_parameter_vec_without -mean_prior)/std_prior
MLE_gradient_and_ABC = (MLE_parameter_vec_with - mean_prior)/std_prior
CI_gradient_only = relative_CIs_without
CI_gradient_and_ABC = relative_CIs_with
np.save('wound_SMC_relative_CIs.npy', CI_gradient_and_ABC*std_prior)


p = len(MLE_gradient_only)
indices = np.arange(p)
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

# Original parameter names converted to numpy array
parameter_names = np.array(parameter_names)

# threshold for splitting
threshold = 3 / (1.66 * 0.05)

# Determine groups
indices_A = np.where(CI_gradient_and_ABC <= threshold)[0]   # small CIs
indices_B = np.where(CI_gradient_and_ABC > threshold)[0]    # large CIs

groups = {
    "Small CIs": indices_A,
    "Large CIs": indices_B,
}

# Create mapping name → list of parameter names
parameter_names_groups = {
    "Small CIs": parameter_names[indices_A],
    "Large CIs": parameter_names[indices_B],
}

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(22, 6), sharey=False)

for ax, (title, idxs) in zip(axes, groups.items()):
    inds = np.arange(len(idxs))

    # Plot true values


    # Gradient-only MLE
    ax.errorbar(
        inds - 0.12,
        MLE_gradient_only[idxs],
        yerr=CI_gradient_only[idxs],
        fmt='o', capsize=3,
        label='MLE (grad only)', zorder=2
    )

    # ABC → gradient MLE
    ax.errorbar(
        inds + 0.12,
        MLE_gradient_and_ABC[idxs],
        yerr=CI_gradient_and_ABC[idxs],
        fmt='s', capsize=3,
        label='MLE (ABC→grad)', zorder=3
    )

    # Labels
    ax.set_title(f"{title} (n={len(idxs)})", fontsize=14)
    ax.set_xticks(inds)
    ax.set_xticklabels(parameter_names_groups[title], rotation=90, fontsize=10)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Parameter value (z-scale)", fontsize=12)
# Collect handles/labels from all axes
handles, labels = [], []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

# Deduplicate while preserving order
unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels))
          if l not in labels[:i]]
unique_handles, unique_labels = zip(*unique)

# Single clean legend for the whole figure
fig.legend(
    unique_handles,
    unique_labels,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
    fontsize=12
)
plt.tight_layout()
plt.show()






colors = ['red', 'green', 'blue', 'orange', 'purple']
color_labels = ['FB-1', 'FB-II', 'FB-III','Mono-Mac', 'EC']

markers = ['o', 's', '^']  # circle, square, triangle
marker_labels = ['TGFB1', 'TGFB2', 'TGFB3']


y = interaction_data.flatten()
x = MLE_instance_with.forward_I_model(MLE_z_tensor_with).cpu().detach().numpy().flatten()

# Plot points
for i in range(75):
    color = colors[i % 5]
    marker = markers[i // 25]
    plt.scatter(x[i], y[i], color=color, marker=marker)

# # Fit line
# coeffs = np.polyfit(x, y, deg=1)
# slope, intercept = coeffs
# plt.plot(x, slope * x + intercept, color='black', linestyle='--', label='Fit')

# Create legend for colors
color_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8)
    for c in colors
]
color_legend = plt.legend(color_handles, color_labels, title='Producing cell', loc='upper left')

# Create legend for markers
marker_handles = [
    plt.Line2D([0], [0], marker=m, color='black', linestyle='None', markersize=8)
    for m in markers
]
plt.legend(marker_handles, marker_labels, title='Ligand/receptor pair', loc='lower right')

# Add color legend back so both show
plt.gca().add_artist(color_legend)

plt.xlabel("Model-derived Interaction Strengths [pmol/s]")
plt.ylabel("Data-derived Interaction Strengths (a.u)")
#plt.title('Interaction Score Correlation')
plt.show()


for i in range(75):
    color = colors[(i%25) // 5]
    marker = markers[i // 25]
    plt.scatter(x[i], y[i], color=color, marker=marker)
color_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8)
    for c in colors
]
color_legend = plt.legend(color_handles, color_labels, title='Receiving cell', loc='upper left')

# Create legend for markers
marker_handles = [
    plt.Line2D([0], [0], marker=m, color='black', linestyle='None', markersize=8)
    for m in markers
]
plt.legend(marker_handles, marker_labels, title='Ligand/receptor pair', loc='lower right')

# Add color legend back so both show
plt.gca().add_artist(color_legend)

plt.xlabel("Model-derived Interaction Strengths [pmol/s]")
plt.ylabel("Data-derived Interaction Strengths (a.u)")
#plt.title('Interaction Score Correlation')
plt.show()



#Side by side for NPJ journal
# --- Create the figure and subplots ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

########################
# ---- PANEL (a) ----  #
########################

ax = axes[0]

for i in range(75):
    color = colors[i % 5]
    marker = markers[i // 25]
    ax.scatter(x[i], y[i], color=color, marker=marker)

# Legends
color_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8)
    for c in colors
]
color_legend = ax.legend(color_handles, color_labels,
                         title='Producing cell', loc='upper left')

marker_handles = [
    plt.Line2D([0], [0], marker=m, color='black', linestyle='None', markersize=8)
    for m in markers
]
ax.legend(marker_handles, marker_labels, title='Ligand/receptor pair',
          loc='lower right')
ax.add_artist(color_legend)

ax.set_xlabel("Model-derived Interaction Strengths [pmol/s]")
ax.set_ylabel("Data-derived Interaction Strengths (a.u.)")
ax.set_title("(a)", fontsize=14, fontweight="bold", pad=10)


########################
# ---- PANEL (b) ----  #
########################

ax = axes[1]

for i in range(75):
    color = colors[(i % 25) // 5]
    marker = markers[i // 25]
    ax.scatter(x[i], y[i], color=color, marker=marker)

# Legends
color_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8)
    for c in colors
]
color_legend = ax.legend(color_handles, color_labels,
                         title='Receiving cell', loc='upper left')

marker_handles = [
    plt.Line2D([0], [0], marker=m, color='black', linestyle='None', markersize=8)
    for m in markers
]
ax.legend(marker_handles, marker_labels, title='Ligand/receptor pair',
          loc='lower right')
ax.add_artist(color_legend)

ax.set_xlabel("Model-derived Interaction Strengths [pmol/s]")
ax.set_ylabel("Data-derived Interaction Strengths (a.u.)")
ax.set_title("(b)", fontsize=14, fontweight="bold", pad=10)

########################
# ---- SAVE FIGURE ----#
########################

plt.tight_layout()
plt.savefig("Figure 8.png", dpi=600, bbox_inches="tight")
plt.close()





