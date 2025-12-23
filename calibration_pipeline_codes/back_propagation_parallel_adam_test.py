#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 14:28:06 2025

@author: raluca
"""

from mpi4py import MPI
from scipy.sparse.linalg import splu
#from scipy.sparse import csc_matrix
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle as pkl
import pandas as pd
import time
from scipy.sparse import csc_matrix
#import matplotlib.pyplot as plt



        
def build_system(theta, cell_densities, mesh_prop, neigbourhood_list):
        
        
        np_cell_densities = cell_densities.cpu().numpy()
        n= cell_densities.shape[1]
        Area, L_c, L_e = mesh_properties
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
        return g, dg_dtheta            
    


class SolveG(torch.autograd.Function):
    
    

    @staticmethod
    def forward(ctx, theta_full, cell_densities, mesh_prop,\
                     neigbourhood_list):
        # Build A, b from theta
        # Also construct dA, db while looping
        theta_np = theta_full.detach().cpu().numpy()
        g_np , dg_dtheta_np = build_system(theta_np,\
                              cell_densities, mesh_prop, neigbourhood_list)
        g = torch.from_numpy(g_np).to(dtype=theta_full.dtype, device=device)
        dg_dtheta = torch.from_numpy(dg_dtheta_np).to(dtype=theta_full.dtype, device=device)
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
        int_str = int_str*1e25
        return int_str
    
    

    
    
class multi_ligand_cob(single_ligand_model):    
    def __init__(self, log_theta, scale,\
                 neighbours_list, cell_densities, mesh_prop, I_exp,\
                     seg_shared, seg_1, seg_2, seg_3, min_vals, max_vals, regularization_param):
        super().__init__(neighbours_list, cell_densities, mesh_prop)
        

        self.I_exp = torch.tensor(I_exp, dtype=torch.float32, device=device)  
        self.scaled_log_theta = nn.Parameter(log_theta)
        self.scale = scale
        self.seg_shared = torch.tensor(seg_shared,dtype=torch.long, device = device)
        self.seg_1 = torch.tensor(seg_1,dtype=torch.long, device = device)
        self.seg_2 = torch.tensor(seg_2,dtype=torch.long, device = device)
        self.seg_3 = torch.tensor(seg_3,dtype=torch.long, device = device)
        self.cell_densities = torch.tensor(cell_densities, dtype=torch.float32, device = device)
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.lambda_r = regularization_param
    
    def construct_thetas_full(self, log_scaled_theta):  
        theta_unscaled = torch.exp(log_scaled_theta+self.scale)
        theta_1 = torch.cat([theta_unscaled[self.seg_shared], theta_unscaled[self.seg_1]])
        theta_2 = torch.cat([theta_unscaled[self.seg_shared], theta_unscaled[self.seg_2]])
        theta_3 = torch.cat([theta_unscaled[self.seg_shared], theta_unscaled[self.seg_3]])
        return theta_1, theta_2, theta_3
    
    def calc_loss_func(self, I_model_1, I_model_2, I_model_3):
        I_model = torch.cat([I_model_1.flatten(), I_model_2.flatten(),I_model_3.flatten()])
        I_data = self.I_exp.flatten()
        covariance_biased = torch.mean(I_model*I_data) - torch.mean(I_model)*torch.mean(I_data)
        std_model_biased = torch.std(I_model, correction = 0)
        std_data_biased = torch.std(I_data, correction = 0)
        correlation = covariance_biased/(std_model_biased*std_data_biased)
        loss = 1-correlation
        return loss
    
    def regularization(self, theta):
        # theta_clipped =  theta#torch.clip(theta, min=0.0)
        # sig1 = torch.sigmoid(self.heaviside_slope * (theta_clipped - self.min_vals/self.scale))
        # sig2 = torch.sigmoid(self.heaviside_slope * (self.max_vals/self.scale - theta_clipped))
        elementwise_regularization = torch.clip(torch.maximum(torch.exp(theta) - self.max_vals/torch.exp(self.scale),\
                                    self.min_vals/torch.exp(self.scale) - torch.exp(theta)), min=0.0)
        return torch.sum(elementwise_regularization)
    
    
    def forward(self):
        
        theta_1, theta_2, theta_3 = self.construct_thetas_full(self.scaled_log_theta)
        g_1 = SolveG.apply(theta_1,self.cell_densities, self.mesh_prop,self.neighbours_list)
        I1 = self.calc_int_str(g_1, theta_1)    
        g_2 = SolveG.apply(theta_2,self.cell_densities, self.mesh_prop,self.neighbours_list)
        I2 = self.calc_int_str(g_2, theta_2) 
        g_3 = SolveG.apply(theta_3,self.cell_densities, self.mesh_prop,self.neighbours_list)
        I3 = self.calc_int_str(g_3, theta_3) 
        loss_A = self.calc_loss_func(I1, I2, I3) 
        loss_B = self.lambda_r*self.regularization(self.scaled_log_theta)
        loss = loss_A +loss_B
        print(f"loss A = {loss_A}")
        print(f"loss B = {loss_B}")
        return loss
    


def train_single_seed(seed, optimize_scheme, theta, neighbours_list,\
                      cell_densities, mesh_prop, I_exp, seg_shared, seg_1, seg_2, seg_3,\
                      min_vals, max_vals, relative_learning_rates,l_r,num_epochs, regularization_param):

      
    
    log_theta = torch.tensor(np.log(theta), dtype = torch.float32, device=device)
    scale = log_theta.detach().clone()
    min_vals = torch.tensor(min_vals, dtype = torch.float32, device=device)
    max_vals = torch.tensor(max_vals, dtype = torch.float32, device=device)
    scaled_var = log_theta - scale
# ---- Training Loop ----
    model = multi_ligand_cob(scaled_var, scale,\
                         neighbours_list, cell_densities, mesh_prop, I_exp,\
                         seg_shared, seg_1, seg_2, seg_3, min_vals, max_vals, regularization_param)
    if optimize_scheme == "Adam":    
        optimizer = optim.Adam(model.parameters(), lr=l_r)
    elif optimize_scheme == "Stochastic Gradient Descent":
        optimizer = optim.SGD(model.parameters(), lr=l_r)
    
    loss_history = []
   # initial_params = torch.exp(scale*{name: param.detach().cpu().clone() for name, param in model.named_parameters()}['scaled_log_theta'])
    
    
    relative_learning_rates = torch.tensor(relative_learning_rates, dtype=torch.float32, device=device)
    start_time = time.time()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = model()
        loss.backward()  # Triggers custom and automatic backward
        #print(f"model.scaled_theta.grad norm: {model.scaled_log_theta.grad.norm().item()}", flush=True)
        with torch.no_grad():
            model.scaled_log_theta.grad *= relative_learning_rates
        optimizer.step()
        loss_history.append(loss.item())
        #print(f"Epoch {epoch}, Loss: {loss.item()}") 
       
    total_time = time.time() - start_time
    final_params = torch.exp(scale+{name: param.detach().cpu().clone() for name, param in model.named_parameters()}['scaled_log_theta']).detach().cpu().numpy()  
    #np.save(f"loss_history_{optimize_scheme}_optimzer_seed_{seed}.npy", np.array(loss_history))
    #torch.save(model.state_dict(), f"model_{optimize_scheme}_seed_{seed}.pt")
    return loss.item(), loss_history, final_params, total_time


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
    file_path = 'synthetic_interaction_strengths.ods'

    # Read each sheet into a dictionary of DataFrames
    sheets = ['TGFB1-TGFBR2', 'TGFB2-TGFBR3', 'TGFB3-TGFBR2']
    dfs = {sheet: pd.read_excel(file_path, sheet_name=sheet, engine='odf') for sheet in sheets}

    # Now, we can extract the 5x5 tables from each sheet into a 3D NumPy array
    # We slice each DataFrame to remove the first row and first column

    interaction_data = np.array([dfs[sheet].iloc[:, 1:].values for sheet in sheets])




    parameter_array = np.load("starting_seeds.npy")
    
    
    return mesh_properties, neighbours, cell_densities, interaction_data, parameter_array

    


# Segment indices #need not to be continuous! segment A for D, lambda, segment B for the rhos and Ks






if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    
    if rank==0:
        print("run is starting! We are at rank 0", flush=True)
        mesh_properties, neighbours, cell_densities, interaction_data, parameter_array = get_constant_input()
        split_indices = np.array_split(np.arange(parameter_array.shape[0]), size)
    else:
        mesh_properties, neighbours, cell_densities, interaction_data, parameter_array = None, None, None, None, None
        split_indices = None
    
    mesh_properties = comm.bcast(mesh_properties, root = 0)
    neighbours = comm.bcast(neighbours, root = 0)
    cell_densities = comm.bcast(cell_densities, root=0)
    interaction_data = comm.bcast(interaction_data, root=0)
    parameter_array = comm.bcast(parameter_array, root=0)
    
    
    seg_A = np.array([0,1])
    seg_1B = np.array([2,5,8,11,14,17,20,23,26,29,32])
    seg_2B = np.array([3,6,9,12,15,18,21,24,27,30,33])
    seg_3B = np.array([4,7,10,13,16,19,22,25,28,31,34])
    #seg 1_B, 2_B, and 3_B refer to ligands (TGFB) 1, 2, and 3.
    seg_B_list = [seg_1B, seg_2B, seg_3B]
    #print('realization segmentation done')


    d=parameter_array.shape[1]

    MC_fixed_param_indices = np.array([0,1,14])
    SMC_param_var_indices = np.array([4,6,7])
    SMC_param_fixed_indices = np.setdiff1d(np.arange(d),np.union1d(MC_fixed_param_indices,SMC_param_var_indices))
    relative_learning_rates = np.zeros(d)
    relative_learning_rates[MC_fixed_param_indices]= 10
    relative_learning_rates[SMC_param_fixed_indices]=5
    relative_learning_rates[SMC_param_var_indices] = 1





    Av = 6.0234e23
    min_vals = np.array(
        [0.5e-11, 1e-6] +
        [1e-24] * 9 +
        [0.5e-24] * 3 +
        [0.2e-24] *3 +
        [3e2,3e2,4e2] +
        [6000 / Av] * 9+
        [300 / Av] * 3 +
        [4000 / Av] * 3
        
    )

    max_vals = np.array(
        [5e-10, 5e-5] +
        [6e-22] * 9 +
        [4e-22] * 3 +
        [3e-22] *3 +
        [9e4,9e4,9e4] +
        [18000 / Av] * 9+
        [10000 / Av] * 3 +
        [15000 / Av] * 3
        
    )
    
    
    
    
    
    
    # Split realizations among MPI ranks
    split_indices = comm.bcast(split_indices, root=0)
    local_indices = split_indices[rank]

    local_seedings = parameter_array[local_indices, :]
    #chunk_size = local_seedings.shape[0]

    basic_learning_rate = [1e-2,2.5e-2,4e-2]
    epochs = int(1e3)
    regularization_param = 0.002#[0.001,0.005, 0.01, 0.015, 0.02]
    local_histories = np.zeros([local_seedings.shape[0],len(basic_learning_rate), epochs])
    local_final_loss = np.zeros([local_seedings.shape[0],len(basic_learning_rate)])
    local_parameters = np.zeros([local_seedings.shape[0],len(basic_learning_rate), d])
    local_running_times = np.zeros([local_seedings.shape[0],len(basic_learning_rate)])
    
    for i in range(local_seedings.shape[0]):
        for j, lr in enumerate(basic_learning_rate):
            #for k, rp in enumerate(regularization_param):
            theta = local_seedings[i,:]
            seed_num = local_indices[i]
            print(f"seed number is {seed_num}", flush=True)
            final_loss, loss_history, final_parameters, running_time =\
            train_single_seed(seed_num, "Adam", theta,\
                             neighbours, cell_densities, mesh_properties,interaction_data,\
                                 seg_A, seg_1B, seg_2B, seg_3B, min_vals, max_vals,\
                                     relative_learning_rates, lr, epochs, regularization_param)
            local_histories[i,j,:] = loss_history
            local_final_loss[i,j] = final_loss
            local_parameters[i,j,:] = final_parameters
            local_running_times[i,j] = running_time

    all_loss = comm.gather(local_final_loss, root=0)
    all_histories = comm.gather(local_histories, root=0)
    all_parameters = comm.gather(local_parameters, root=0)
    all_running_times = comm.gather(local_running_times)
    
    comm.Barrier()
    if rank==0:
        counter=0
        final_histories = np.zeros([parameter_array.shape[0], len(basic_learning_rate), epochs ])
        final_loss = np.zeros([parameter_array.shape[0], len(basic_learning_rate)])
        final_parameter_values = np.zeros([parameter_array.shape[0], len(basic_learning_rate), d])
        final_running_times = np.zeros([parameter_array.shape[0], len(basic_learning_rate)])
        for i in range(len(all_histories)):
            hist_a = all_histories[i]
            loss_a = all_loss[i]
            parameters_a = all_parameters[i]
            running_times_a = all_running_times[i]
            local_size = hist_a.shape[0]
            final_histories[counter:counter+local_size,:,:] = hist_a
            final_loss[counter:counter+local_size,:] = loss_a
            final_parameter_values[counter:counter+local_size,:,:] = parameters_a
            final_running_times[counter:counter+local_size,:] = running_times_a
            counter+=local_size

        with open(f"final_ML_results_{parameter_array.shape[0]}_seeds_{epochs}_epochs_ADAM.pkl", "wb") as f:
            pkl.dump({"loss": final_loss, "parameters": final_parameter_values,\
                         "histories": final_histories, "running_times": final_running_times,\
                             "learning_rates": basic_learning_rate,"regularization_params": regularization_param}, f)
        




