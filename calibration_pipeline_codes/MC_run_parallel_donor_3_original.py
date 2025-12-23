#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 13:41:43 2025

@author: ali
"""

from mpi4py import MPI
import numpy as np
from multiprocessing import Pool 
from functools import partial
import pickle
import pandas as pd
import os
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix



# ----------------------
#  functions
# ----------------------



def FVM_steady_state_ligand_con(neigbourhood_list, mesh_properties, param_A, param_B, cell_densities):
    """
    solve for the growth factor/ligand concentration at the FVM centres

    Inputs:

    cell_densities: an N_CV by n array, where N_CV is number of CVs, n is number of
    cell types if interest.

    neighbourhood list: a list of arrays describing the neigbours for each CV
    mesh properties: Area of CV, L_c: distance between centroids, L_e: length of edge
    parameter_vector: D: diffusion coefficient, lambda_g: decay constant,
    rho: production rates (of size n),
    K: association constant (scaled with receptor number) (of size n)

    Output: concentration of ligand at the CV (assuming steady-state)
    """
    N_CV = cell_densities.shape[0]
    n= cell_densities.shape[1]
    Area, L_c, L_e = mesh_properties
    D, lambda_g = param_A
    rho = param_B[:n]
    K = param_B[n]
    rec_number = param_B[-n:]

    A = np.zeros((N_CV, N_CV))
    b = np.zeros((N_CV))
    for i in range(N_CV):
        neighbours = neigbourhood_list[i]
        lig_prod = cell_densities[i,:]@rho
        lig_absor = cell_densities[i,:]@(rec_number*K)
        A[i,i] = D*L_e/L_c * len(neighbours) + Area*(lambda_g+lig_absor)
        A[i, neighbours] = -D*L_e/L_c
        b[i] = Area*lig_prod
    A = csr_matrix(A*1e11)
    b= b*1e23
    g = spsolve(A,b)
    return g #in pmol/m^2

def calc_interaction_strength(cell_densities, g_conc, param):

    N_CV = cell_densities.shape[0]
    n = cell_densities.shape[1]

    rho = param[:n]
    K = param[n]
    rec_number = param[-n:]  #size n



    num_1 = cell_densities*rho   #Size CV by n
    denom_1 = np.sum(num_1, axis=1)  #size CV
    prod_term = num_1/denom_1[:,None]  #size CV by n

    absor_array = K*rec_number[:,None] * g_conc[None,:] #size n by CV: outer product
    interaction_strength_array = 	np.matmul(absor_array, prod_term)
    #where first dimensions (rows) correspond the receiving cells and second dimension
    #(columns) refers to producing cells
    return 1/N_CV*interaction_strength_array



def compute_segment(segB: np.ndarray,r_i: np.ndarray, segA: np.ndarray, neighbours, mesh_prop, cell_dens) -> float:
    g =  FVM_steady_state_ligand_con(neighbours, mesh_prop, r_i[segA], r_i[segB], cell_dens)
    y = calc_interaction_strength(cell_dens,g,r_i[segB])
    return y

def calc_correlation(I1, I2, I3, I_data):
    """
    Inputs:
    I1, I2, and I3 are for g1 (TGFB1), g2 (TGFB2) and g3 (TGFB3): rows are receiving cells
    #and columns are g producing cells

    interaction_strength_data: n by n by #ligands (3) here; each slice corresponds to a ligand


    #both data from model and scRNA-seq are arranged such that rows are receiving cells and
    #columns are ligand producing cells

    Output: float number signifiying correlation coefficient

    """

    I_model = np.concatenate((I1.flatten(), I2.flatten(), I3.flatten()))
    I_data = I_data.flatten()

    return np.corrcoef(I_model, I_data)[0, 1]



# ----------------------
# Per-realization evaluation
# ----------------------
def get_constant_inputs():
    
    """
    Define variables used for all MC simulations
    """

    L_c = 84.5e-6 #from smallest distance
    hotspot_area = np.sqrt(3)/2*L_c**2
    L_e = L_c/np.sqrt(3)

    mesh_properties = [hotspot_area,L_c,L_e]
    with open('dataframe.pkl', 'rb') as file:
        neighbours_info = pickle.load(file)
        neighbours=neighbours_info.iloc[:, 1].tolist()

    q05_cell_numbers = pd.read_csv('q05_cell_abundances.csv')
    q05_cell_numbers.columns.values[0] = 'Nucleotide_ID'
    are_identical = neighbours_info['ID'].equals(q05_cell_numbers['Nucleotide_ID'])
    print(f"Are the columns identical? {are_identical}", flush=True)


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
    
    return mesh_properties, cell_densities, neighbours, interaction_data




def construct_MC_realization():

    """Monte Carlo Configuration"""
    #-----------------------------------
    #Monte Carlo Filtering configuration
    #-----------------------------------

    MC = int(3e5)
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





    realizations = np.column_stack([D, lambda_g, rho_fib1_1, rho_fib1_2, rho_fib1_3, \
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
    
    
    


# Segment indices #need not to be continuous! segment A for D, lambda, segment B for the rhos and Ks
    seg_A = np.array([0,1])
    seg_1B = np.array([2,5,8,11,14,17,20,23,26,29,32])
    seg_2B = np.array([3,6,9,12,15,18,21,24,27,30,33])
    seg_3B = np.array([4,7,10,13,16,19,22,25,28,31,34])
    #seg 1_B, 2_B, and 3_B refer to ligands (TGFB) 1, 2, and 3.
    seg_B_list = [seg_1B, seg_2B, seg_3B]
    print('realization segmentation done') 
    
    return realizations, seg_A, seg_B_list



def compute_score(r):
    evaluate_partial = partial(compute_segment,r_i=r,segA=seg_A,neighbours=neighbours,
                               mesh_prop=mesh_properties,cell_dens=cell_densities)
    
    with Pool(processes=num_workers) as p:
        I_S = p.map(evaluate_partial, seg_B_list)
    correlation_coeff = calc_correlation(I_S[0], I_S[1], I_S[2], interaction_data)
    return correlation_coeff

   

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 3))
    epsilon = 0.2
    if rank ==0:
        mesh_properties, cell_densities, neighbours, interaction_data =get_constant_inputs()
        realizations, seg_A, seg_B_list = construct_MC_realization()
    else:
        mesh_properties, cell_densities, neighbours, interaction_data = None, None, None, None
        realizations, seg_A, seg_B_list = None, None, None
        
    cell_densities = comm.bcast(cell_densities, root=0)
    mesh_properties = comm.bcast(mesh_properties, root=0)
    neighbours = comm.bcast(neighbours, root=0)
    interaction_data = comm.bcast(interaction_data, root=0)
    
    realizations = comm.bcast(realizations, root=0)
    MC = realizations.shape[0]
    seg_A = comm.bcast(seg_A, root=0)
    seg_B_list = comm.bcast(seg_B_list, root=0)
    
    
    # Split realizations among MPI ranks
    chunk_size = MC // size
    start = rank * chunk_size
    end = (rank + 1) * chunk_size if rank != size - 1 else MC
    local_realizations = realizations[start:end]
    
    local_accepted_scores = []
    local_accepted_realizations=[]
    print(f"starting for rank {rank}", flush=True)
    for i in range(local_realizations.shape[0]):
        realization = local_realizations[i,:]
        score = compute_score(realization)
        if score>epsilon:
            local_accepted_scores.append(score)
            local_accepted_realizations.append(realization)
     
    # Gather results from all MPI processes        
    all_scores  = comm.gather(local_accepted_scores, root=0)
    all_realizations = comm.gather(local_accepted_realizations,root=0)
    
    if rank==0:
        final_scores =[]
        final_accepted_realizations = []
        for index, item in enumerate(all_scores):
            item_r = all_realizations[index]
            for index_2, item_2 in enumerate(item_r):
                final_scores.append(item[index_2])
                final_accepted_realizations.append(item_r[index_2])
 
        with open("accepted_results.pkl", "wb") as f:
            pickle.dump({"scores": final_scores, "realizations": final_accepted_realizations}, f)