#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 16:12:29 2025

@author: ali
"""

from scipy.sparse.linalg import splu
from scipy.sparse import csr_matrix
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle as pkl
import pandas as pd
import sys



        
def build_system(theta_full, var_param_indices, cell_densities, mesh_prop,\
                 neigbourhood_list):
        
        np_cell_densities = cell_densities.cpu().numpy()
        D, lambda_g = theta_full[0], theta_full[1]
        rho = theta_full[2:7]
        K = theta_full[7]
        rec_number = theta_full[8,:]
        CV_num = cell_densities.shape[0]
        area, L_c, L_e = mesh_prop
        L_ratio = L_e/L_c
        A = np.zeros((CV_num, CV_num))
        b = np.zeros((CV_num))
        dA_dtheta = np.zeros((CV_num,CV_num,len(theta_full)))
        db_dtheta = np.zeros((CV_num, len(theta_full)))
        dg_dtheta = np.zeros((len(var_param_indices), CV_num))
        for i in range(CV_num):
            neighbours = neigbourhood_list[i]
            A[i,i] = D*L_ratio*len(neighbours)+area*(lambda_g +\
                     np_cell_densities[i,:]@(rec_number*K))
            b[i] = area * np_cell_densities[i,:]@rho
            A[i, neighbours] = -D*L_ratio
            db_dtheta[i,2:7] = area * np_cell_densities[i,:]
            dA_dtheta[i, neighbours,0] = -L_ratio
            dA_dtheta[i,i,0] = L_ratio*len(neighbours)
            dA_dtheta[i,i,1] = area
            dA_dtheta[i,i,7] = area*np_cell_densities[i,:]@rec_number
            dA_dtheta[i,i,8:] = area*np_cell_densities[i,:]*K
        
        A = csr_matrix(A * 1e11)
        b *= 1e23
        dA_dtheta *= 1e11
        db_dtheta *= 1e23
        lu = splu(A)
        g = lu.solve(b) #in pmol/m^2
        for idx_1, idx_2 in enumerate(var_param_indices):
            RHS = db_dtheta[:,idx_2]- dA_dtheta[:,:,idx_2]@g
            dg_dtheta[idx_1,:] = lu.solve(RHS)
        return g, dg_dtheta            
    


class SolveG(torch.autograd.Function):
    
    

    @staticmethod
    def forward(ctx, theta_full, var_param_indices, cell_densities, mesh_prop,\
                     neigbourhood_list):
        # Build A, b from theta
        # Also construct dA, db while looping
        
        theta_np = theta_full.detach().cpu().numpy()
        g_np , dg_dtheta_np = build_system(theta_np, var_param_indices,\
                              cell_densities, mesh_prop, neigbourhood_list)
        g = torch.from_numpy(g_np).to(dtype=theta_full.dtype, device=device)
        dg_dtheta = torch.from_numpy(dg_dtheta_np).to(dtype=theta_full.dtype, device=device)
        ctx.save_for_backward(dg_dtheta)
        return g
    @staticmethod
    def backward(ctx, grad_output):
        dg_dtheta, = ctx.saved_tensors  # shape (N_var, M)
        grad_output = grad_output.view(-1)  # shape: (M,)
        grad_theta = torch.matmul(dg_dtheta, grad_output)  # shape (N,)
        return grad_theta, None, None, None, None
    
    



    
# Your main model (could be more complex)
class single_ligand_model(nn.Module):
    def __init__(self, neighbours_list, cell_densities, mesh_prop):
        super().__init__()
        # θ is the learnable parameter
        
        self.cell_densities = torch.tensor(cell_densities, dtype=torch.float32, device=device)
        self.neighbours_list = neighbours_list
        self.CV_num = cell_densities.shape[0]
        self.mesh_prop = mesh_prop


        

    
    def calc_int_str(self,g,full_theta):
        rho = full_theta[2:7]
        K = full_theta[7]
        rec_number = full_theta[8:]
        
        num_1 = self.cell_densities*rho   #Size CV by n
        denom_1 = torch.sum(num_1, axis=1)  #size CV
        prod_term = num_1 / denom_1.unsqueeze(1) #size CV by n
        absorb_array = K *torch.outer(rec_number, g) #size n by CV
        int_str = 1/self.CV_num*absorb_array@prod_term #n by n
        return int_str
    
    

    
    
class multi_ligand_cob(single_ligand_model):    
    def __init__(self, theta_var_1, theta_var_2, theta_var_3, theta_var_ind_1,\
                 theta_var_ind_2, theta_var_ind_3, theta_fixed_1, theta_fixed_2, theta_fixed_3,\
                 neighbours_list, cell_densities, mesh_prop, I_exp):
        super().__init__(neighbours_list, cell_densities, mesh_prop)
        
        self.d = len(theta_var_1) + len(theta_fixed_1)
        self.I_exp = torch.tensor(I_exp, dtype=torch.float32, device=device)
        
        self.register_buffer('fixed_param_1', torch.tensor(theta_fixed_1, dtype=torch.float32, device=device))
        self.register_buffer('fixed_param_2', torch.tensor(theta_fixed_2, dtype = torch.float32, device=device))
        self.register_buffer('fixed_param_3', torch.tensor(theta_fixed_3, dtype=torch.float32,device=device))
        
        self.var_param_indices_1 = torch.tensor(theta_var_ind_1, dtype=torch.long, device=device)
        self.theta_var_1 = nn.Parameter(torch.tensor(theta_var_1, dtype=torch.float32, device=device))
       
        
        self.var_param_indices_2 = torch.tensor(theta_var_ind_2, dtype=torch.long, device=device)
        self.theta_var_2 = nn.Parameter(torch.tensor(theta_var_2, dtype=torch.float32, device=device))
       
        
        self.var_param_indices_3 = torch.tensor(theta_var_ind_3, dtype=torch.long, device=device)
        self.theta_var_3 = nn.Parameter(torch.tensor(theta_var_3, dtype=torch.float32, device=device))    
        

    def construct_theta_full(self, theta_fixed_param, theta_var_param, var_param_indices):
        out = torch.zeros(self.d, dtype=torch.float32, device=device)
        theta_partial = out.scatter(0, var_param_indices, theta_var_param)
        fixed_param_indices = torch.tensor(np.setdiff1d(np.arange(self.d), 
                             var_param_indices.cpu().numpy()),\
                              dtype=torch.long, device=device)
        theta_full = theta_partial.scatter(0, fixed_param_indices,theta_fixed_param)    
        return theta_full
    
    def calc_loss_func(self, I_model_1, I_model_2, I_model_3):
        I_model = torch.cat([I_model_1.flatten(), I_model_2.flatten(),I_model_3.flatten()])
        I_data = self.I_exp.flatten()
        covariance_biased = torch.mean(I_model*I_data) - torch.mean(I_model)*torch.mean(I_data)
        std_model_biased = torch.std(I_model, correction = 0)
        std_data_biased = torch.std(I_data, correction = 0)
        loss = 1-covariance_biased/(std_model_biased*std_data_biased)
        return loss
    
    
    def forward(self):
        
        theta_full_1 = self.construct_theta_full(self.fixed_param_1, self.theta_var_1,\
                                self.var_param_indices_1)
        g_1 = SolveG.apply(theta_full_1, self.var_param_indices_1,\
                           self.cell_densities, self.mesh_prop,self.neighbours_list)    
        I1 = self.calc_int_str(g_1, theta_full_1)    
            
            
        theta_full_2 = self.construct_theta_full(self.fixed_param_2, self.theta_var_2,\
                                self.var_param_indices_2)
        g_2 = SolveG.apply(theta_full_2, self.var_param_indices_2,\
                           self.cell_densities, self.mesh_prop,self.neighbours_list)
        I2 = self.calc_int_str(g_2, theta_full_2)                          
            
            
            
            
        theta_full_3 = self.construct_theta_full(self.fixed_param_3, self.theta_var_3,\
                                    self.var_param_indices_3)
        
        g_3 = SolveG.apply(theta_full_3, self.var_param_indices_3,\
                               self.cell_densities, self.mesh_prop,self.neighbours_list)    
        I3 = self.calc_int_str(g_3, theta_full_3)    
            


        loss = self.calc_loss_func(I1, I2, I3)
        return loss
    


def train_single_seed(seed, optimize_scheme, theta_var_1, theta_var_2, theta_var_3,\
                      theta_var_ind_1,theta_var_ind_2, theta_var_ind_3,\
                     theta_fixed_1, theta_fixed_2, theta_fixed_3,\
                     neighbours_list, cell_densities, mesh_prop, I_exp, l_r=1e-2,num_epochs=100):
# ---- Training Loop ----
    model = multi_ligand_cob(theta_var_1, theta_var_2, theta_var_3,\
                          theta_var_ind_1,theta_var_ind_2, theta_var_ind_3,\
                         theta_fixed_1, theta_fixed_2, theta_fixed_3,\
                         neighbours_list, cell_densities, mesh_prop, I_exp)
    if optimize_scheme == "Adam":    
        optimizer = optim.Adam(model.parameters(), lr=l_r)
    elif optimize_scheme == "Stochastic Gradient Descent":
        optimizer = optim.SGD(model.parameters(), lr=l_r)
    
    loss_history = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = model()
        loss.backward()  # Triggers custom and automatic backward
        optimizer.step()
        loss_history.append(loss.item())
        print(f"Epoch {epoch}, Loss: {loss.item()}") 
        if (epoch+1)%5 ==0:
            torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
    }, f"checkpoint_{optimize_scheme}_seed_{seed}.pt")  # saves every epoch
        
    np.save(f"loss_history_{optimize_scheme}_optimzer_seed_{seed}.npy", np.array(loss_history))
    torch.save(model.state_dict(), f"model_{optimize_scheme}_seed_{seed}.pt")
    return loss.item()



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
print("Are the columns identical?", are_identical)
cells_of_interest = ('FB-I', 'FB-II', 'FB-III', 'Mono-Mac', 'VE')
# Create new tuple with prefixes
prefix = 'q05cell_abundance_w_sf_'
column_names = tuple(f"{prefix}{cell}" for cell in cells_of_interest)
# Extract those columns
cell_numbers_of_interest = q05_cell_numbers[list(column_names)].copy()
# Rename columns to the original short names
cell_numbers_of_interest.columns = cells_of_interest
cell_densities = cell_numbers_of_interest.to_numpy()/hotspot_area

###Prepare the interaction strength data as 3D array, rows are receiving cells, 
#columns are ligand producing cells, and 3rd dimension is for the different ligands
file_path = 'donor 3 day 30 filtered tgfb data.ods'
# Read each sheet into a dictionary of DataFrames
sheets = ['TGFB1-TGBFR2', 'TGFB3-TGFBR2', 'TGFB2-TGFBR3']
dfs = {sheet: pd.read_excel(file_path, sheet_name=sheet, engine='odf') for sheet in sheets}
# Now, we can extract the 5x5 tables from each sheet into a 3D NumPy array
# We slice each DataFrame to remove the first row and first column
interaction_data = np.array([dfs[sheet].iloc[:, 1:].values for sheet in sheets])





# Segment indices #need not to be continuous! segment A for D, lambda, segment B for the rhos and Ks
seg_A = np.array([0,1])
seg_1B = np.array([2,5,8,11,14,17,20,23,26,29,32])
seg_2B = np.array([3,6,9,12,15,18,21,24,27,30,33])
seg_3B = np.array([4,7,10,13,16,19,22,25,28,31,34])


seed_num= int(sys.argv[1]) #0 # Get the array task ID
parameter_array = np.load("SMC_parameters.npy")
binary_array = np.load("SMC_binary.npy") #size n by d where n is seed number, 
#value of 1 for variable parameters
intialization_full = parameter_array[seed_num]
binary_full = binary_array[seed_num] #load these


first_indices = np.concatenate((seg_A, seg_1B))
second_indices = np.concatenate((seg_A, seg_2B))
third_indices = np.concatenate((seg_A, seg_3B))

theta_1_full = intialization_full[first_indices]
binary_1 = binary_full[first_indices]
theta_var_1_indices = np.where(binary_1==1)[0]
theta_var_1 = theta_1_full[theta_var_1_indices]
theta_fixed_1 = theta_1_full[binary_1==0]

theta_2_full = intialization_full[second_indices]
binary_2 = binary_full[second_indices]
theta_var_2_indices = np.where(binary_2==1)[0]
theta_var_2 = theta_2_full[theta_var_2_indices]
theta_fixed_2 = theta_2_full[binary_2==0]

theta_3_full = intialization_full[third_indices]
binary_3 = binary_full[third_indices]
theta_var_3_indices = np.where(binary_3==1)[0]
theta_var_3 = theta_3_full[theta_var_3_indices]
theta_fixed_3 = theta_3_full[binary_3==0]


final_loss = train_single_seed(seed_num, "Adam", theta_var_1, theta_var_2, theta_var_3,\
                      theta_var_1_indices,theta_var_2_indices, theta_var_3_indices,\
                     theta_fixed_1, theta_fixed_2, theta_fixed_3,\
                     neighbours, cell_densities, mesh_properties,\
                    interaction_data, 1e-2,100)