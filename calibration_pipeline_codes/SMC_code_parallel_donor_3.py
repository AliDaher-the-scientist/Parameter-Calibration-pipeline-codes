#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 12:20:26 2025

@author: Aloush97
"""
from mpi4py import MPI
import os
import numpy as np
#from functools import partial
#from typing import Optional, Tuple
import pickle as pkl
import pandas as pd
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from multiprocessing import Pool
from functools import partial
import sys
import warnings
print(f"Rank {MPI.COMM_WORLD.Get_rank()} running with MPI vendor: {MPI.get_vendor()}", file=sys.stderr)
#------------------------
#checpoints functions
#------------------------

warnings.filterwarnings("error", category=RuntimeWarning)
# Specify checkpoint file paths
checkpoint_file_realizations = "checkpoint_realizations.pkl"
checkpoint_file_scores = "checkpoint_scores.pkl"
checkpoint_file_index = "checkpoint_index.pkl"
checkpoint_file_epsilons = "checkpoint_epsilons.pkl"

def atomic_save(obj, filename):
    temp_filename = filename + ".tmp"
    with open(temp_filename, 'wb') as f:
        pkl.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())  # ensures it's written to disk
    os.replace(temp_filename, filename)  # atomic operation on most OSes

def save_checkpoint(realizations, scores, index, epsilon_vector):
    """
    Saves the current realizations, scores, and the index to checkpoint files atomically.
    """
    atomic_save(realizations, checkpoint_file_realizations)
    atomic_save(scores, checkpoint_file_scores)
    atomic_save(index, checkpoint_file_index)
    atomic_save(epsilon_vector, checkpoint_file_epsilons)

    print(f"Checkpoint saved to {checkpoint_file_realizations}")




def load_checkpoint():
    """
    Loads the checkpointed realizations, scores, and index if they exist.
    Returns None if no checkpoint is found.
    """
    if os.path.exists(checkpoint_file_realizations) and \
       os.path.exists(checkpoint_file_scores) and \
       os.path.exists(checkpoint_file_index) and \
       os.path.exists(checkpoint_file_epsilons):
        with open(checkpoint_file_realizations, 'rb') as f:
            realizations = pkl.load(f)

        with open(checkpoint_file_scores, 'rb') as f:
            scores = pkl.load(f)

        with open(checkpoint_file_index, 'rb') as f:
            index = pkl.load(f)

        with open(checkpoint_file_epsilons, 'rb') as f:
                epsilon_vector = pkl.load(f)

        return realizations, scores, index, epsilon_vector
    else:
        return None, None, 0, None  # No checkpoint found, start from the beginning







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




def get_constant_inputs():

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
    print("Are the columns identical?", are_identical)


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



    seg_A = np.array([0,1])
    seg_1B = np.array([2,5,8,11,14,17,20,23,26,29,32])
    seg_2B = np.array([3,6,9,12,15,18,21,24,27,30,33])
    seg_3B = np.array([4,7,10,13,16,19,22,25,28,31,34])
    #seg 1_B, 2_B, and 3_B refer to ligands (TGFB) 1, 2, and 3.
    seg_B_list = [seg_1B, seg_2B, seg_3B]
    print('realization segmentation done', flush = True)
    
    rec_const = 6.0214*10**23 #avocadro's constant

    prior_list = [ [-11,np.log10(1.3)-10,True], 
                   [4.1e-6, 1.332e-5, False],
                   [1.556e-22, 3.472e-22, False],
                   [1.556e-22, 3.472e-22, False],
                   [1.556e-22, 3.472e-22, False],
                   [0.5*1.556e-22, 0.5*3.472e-22, False],
                   [0.2*1.556e-22, 0.2*3.472e-22, False],
                   [np.log10(5.2)+2, np.log10(3.1)+4, True],
                   [np.log10(5)+2, np.log10(2)+4, True],
                   [np.log10(6.9)+2, np.log10(8.3)+4, True],
                   [8350/rec_const, 14850/rec_const, False],
                   [8350/rec_const, 14850/rec_const, False],
                   [8350/rec_const, 14850/rec_const, False],
                   [(np.log10(3.8)+2)-np.log10(rec_const), (np.log10(8)+3)-np.log10(rec_const), True],
                   [6000/rec_const, 13200/rec_const, False]]
    
    prior_range = [ [1e-11,1.0e-10], 
                   [4.1e-6, 1.332e-5],
                   [1.556e-22, 3.472e-22],
                   [1.556e-22, 3.472e-22],
                   [1.556e-22, 3.472e-22],
                   [0.5*1.556e-22, 0.5*3.472e-22],
                   [0.2*1.556e-22, 0.2*3.472e-22],
                   [5.2e2, 3.1e4],
                   [5.2e2, 2e4],
                   [6.9e2, 8.3e4],
                   [8350/rec_const, 14850/rec_const],
                   [8350/rec_const, 14850/rec_const],
                   [8350/rec_const, 14850/rec_const],
                   [(3.8e2)/rec_const, (8e3)/rec_const],
                   [6000/rec_const, 13200/rec_const]]
    
    
    
    return mesh_properties, cell_densities, neighbours, interaction_data, seg_A, seg_B_list, prior_list, prior_range

# ----------------------
# Segment computation
# ----------------------
def compute_segment(segB: np.ndarray,r_i: np.ndarray, segA: np.ndarray, neighbours, mesh_prop, cell_dens) -> float:
    g =  FVM_steady_state_ligand_con(neighbours, mesh_prop, r_i[segA], r_i[segB], cell_dens)
    y = calc_interaction_strength(cell_dens,g,r_i[segB])
    return y




def mvn_pdf_vectorized(x, mean, cov):
    """
    x: (N, d) points where to evaluate PDF
    mean: (d,) mean vector
    cov: (d, d) covariance matrix
    Returns: (N,) array of pdf values
    """
    d = mean.size
    L = np.linalg.cholesky(cov)  # (d, d)

    diff = x - mean  # (N, d)

    # Solve L y = diff.T for y: y = L^{-1} (x - mean)
    # diff.T shape: (d, N), solve for all points at once
    y = np.linalg.solve(L, diff.T)  # (d, N)

    # Compute squared Mahalanobis distance for each point
    maha_sq = np.sum(y**2, axis=0)  # (N,)

    # Compute normalization constant
    norm_const = 1.0 / (np.power(2 * np.pi, d / 2) * np.prod(np.diag(L)))

    return norm_const * np.exp(-0.5 * maha_sq)





def compute_score(r, epsilon):

    evaluate_partial = partial(compute_segment,r_i=r,segA=seg_A,neighbours=neighbours,
                               mesh_prop=mesh_properties,cell_dens=cell_densities)

    with Pool(processes=num_workers) as p:
        I_S = p.map(evaluate_partial, seg_B_list)
    correlation_coeff = calc_correlation(I_S[0], I_S[1], I_S[2], interaction_data)
    if correlation_coeff>=epsilon:
        return correlation_coeff
    #print(correlation_coeff)
    return None

    # Directly compute the three segments without inner pool



def check_in_range(theta_true, prior_range):
    
    in_range = True
    combined_parameter_values = [theta_true[0], theta_true[1],
                                 np.sum(theta_true[2:5]),
                                 np.sum(theta_true[5:8]),
                                 np.sum(theta_true[8:11]),
                                 np.sum(theta_true[11:14]),
                                 np.sum(theta_true[14:17]),
                                 theta_true[17], theta_true[18], theta_true[19],
                                 np.sum(theta_true[20:23]),
                                 np.sum(theta_true[23:26]),
                                 np.sum(theta_true[26:29]),
                                 np.sum(theta_true[29:32]),
                                 np.sum(theta_true[32:35])
                                 ]
                                 
                           
    for i, el in enumerate(prior_range):
        min_value = el[0]
        max_value = el[1]
        value = combined_parameter_values[i]
        if (value<min_value or value>max_value):
            in_range = False
            break
                  
    return in_range

def pdf_prior(log_theta_all_scaled, prior_list, mean_vec, std_vec,  d_total):
    
    uni_pdf = []
    retreivable_pdfs_indices = [0,1,17,18,19]
   #variable_retreivable_pdfs_indices = list(set(variable_indices) & set(retreivable_pdfs_indices))
    retreivable_pdf_list_indices = [0,1,7,8,9]
   # variable_list_indices = [i for i, x in enumerate(retreivable_pdfs_indices) if x in variable_retreivable_pdfs_indices]
    #retreivable_pdf_list_indices_variable = [retreivable_pdf_list_indices[i] for i in variable_list_indices]
    for i , elem in enumerate(retreivable_pdfs_indices):
        value = log_theta_all_scaled[elem]
        mean = mean_vec[elem]
        std = std_vec[elem]
        
        min_value = prior_list[retreivable_pdf_list_indices[i]][0]
        max_value = prior_list[retreivable_pdf_list_indices[i]][1]
        is_log_uniform = prior_list[retreivable_pdf_list_indices[i]][-1]
        uni_pdf.append(pdf_single(min_value,max_value,is_log_uniform, std,mean,value))
     
    special_list = 13
    special_indices = [29,30,31]
    special_sum = 0
    for i, elm in enumerate(special_indices):
        #if elm in variable_indices:
        special_sum += np.exp(std_vec[elm]*log_theta_all_scaled[elm] + mean_vec[elm])
        #else:
            #special_sum += log_theta_all_scaled[elm]
    min_special = prior_list[special_list][0]
    max_special = prior_list[special_list][1]
    pdf_special = pdf_single(min_special,max_special,True, 1,0,np.log(special_sum))
    
    
        
    
        
    return np.prod(np.array(uni_pdf))*pdf_special
        



def iterative_theta_sample(weights_old, theta_t_minus, kernel_cov_matrix,
                           std_prior, mean_prior, epsilon, prior_list, prior_range):

    "Provides single realization output"
    unnormalized_sample = np.zeros(d_total)
    #unnormalized_sample[fixed_indices] = fixed_MLE
    #variable_indices = np.setdiff1d(np.arange(d_total), fixed_indices)
    attempts=0
    score = None
    while (score is None) and attempts<=max_attempts:
        sample_number = np.random.choice(list(range(N)), p=weights_old) #make sure it is with replacement
        #Apply perturbation kernel
        theta_star =  np.random.multivariate_normal(theta_t_minus[sample_number,:],\
                                     kernel_cov_matrix)#apply pertrubration kernel
        #check if pertrubed realization is inside prior pdf
        #evaluate prior pdf of theta_star and check realization is in prior
        unnormalized_sample = np.exp(std_prior*theta_star+mean_prior)     
        if check_in_range(unnormalized_sample, prior_range):
            score = compute_score(unnormalized_sample, epsilon)
        attempts+=1

    transition_vector = mvn_pdf_vectorized(theta_t_minus,theta_star, kernel_cov_matrix)
        #backward kernel evaluation from theta_star to previous particle realizations
    #mean_vec = np.zeros(d_total)
    #std_vec = np.ones(d_total)
    
    #mean_vec[variable_indices] = mean_prior
    #std_vec[variable_indices] = std_prior
    prior_pdf_i = pdf_prior(theta_star, prior_list, mean_prior, std_prior , d_total)   
    weight_new = prior_pdf_i/np.dot(weights_old, transition_vector)

    if score is None:
        #raise RuntimeError(f"Failed to get valid score for T=0 after {max_attempts} attempts.")
        print(f"failed to get valid score after {max_attempts} attempts; returning zero weight")
        weight_new = 0

    return theta_star, score, weight_new

               
               
              

def pdf_single(a,b,is_log_uniform, std,mean,y):
    
   
    
    x = np.exp(std*y+mean)
    if is_log_uniform:
        indicator_function =  int(x>=10**a and x<=10**b)
        pdf_x = 1/(np.log(10)*(b-a)*x)*indicator_function
    else:
        indicator_function = int(x>=a and x<=b)
        pdf_x = 1/(b-a)*indicator_function
    dxdy = std*x
    pdf_y = pdf_x*dxdy
    return pdf_y





if __name__ == "__main__":

    
    T = int(100)
    N = int(1e4) #number of particles
    #cov_scaling_constant=0.3
    max_attempts = int(2e4)
    adaptive_percentile = 0.2
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 3))

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    chunk_size = N // size
    start = rank * chunk_size
    end = (rank + 1) * chunk_size if rank != size - 1 else N


    if rank ==0:
        mesh_properties, cell_densities, neighbours, interaction_data, seg_A, seg_B_list, prior_list, prior_range\
            =get_constant_inputs()
            
        population_scores = np.zeros([T,N])
        epsilon_vector = np.zeros(T)

            #different processing for T=0
        
        with open("prior_saved_data.pkl", "rb") as f:
            loaded_data = pkl.load(f)
        #fixed_param_indices = loaded_data["fixed_parameter_indices"]
        mean_prior = loaded_data["variable_parm_means"]
        std_prior = loaded_data["variable_parm_stds"]
        #gmm_prior = loaded_data["prior_gmm"]
        #fixed_param_MLE = loaded_data["fixed_variable_MLE"]
        #log_pdf_max = loaded_data["log_pdf_max"]
        theta_t_minus = loaded_data['initial_particles']
        initial_scores= loaded_data['initial_scores']
        population_scores[0,:] = initial_scores
        epsilon_vector[0] = np.percentile(initial_scores, adaptive_percentile)
        d_var = len(mean_prior)
        particle_realizations = np.zeros((T,N,d_var))        
        particle_realizations[0,:,:] = theta_t_minus
        weights_old= 1/N *np.ones(N)      
        kernel_cov_matrix =  np.cov(theta_t_minus,rowvar=False)
        cov_scaling_constant = 1/np.max(std_prior*np.sqrt(np.diag(kernel_cov_matrix)))*np.log(2)
        kernel_cov_matrix = kernel_cov_matrix*(cov_scaling_constant)**2+ \
                              1e-10 * np.eye(d_var)
        
        
        
    else:
        mesh_properties, cell_densities, neighbours, interaction_data, seg_A, seg_B_list, prior_list, prior_range =\
            None, None, None, None, None, None, None, None
        mean_prior, std_prior =\
            None, None
        population_scores, epsilon_vector, particle_realizations = None, None, None
        theta_t_minus, weights_old, kernel_cov_matrix, epsilon_t = None, None, None, None



    cell_densities = comm.bcast(cell_densities, root=0)
    mesh_properties = comm.bcast(mesh_properties, root=0)
    neighbours = comm.bcast(neighbours, root=0)
    interaction_data = comm.bcast(interaction_data, root=0)
    seg_A = comm.bcast(seg_A, root=0)
    seg_B_list = comm.bcast(seg_B_list, root=0)
    prior_list = comm.bcast(prior_list, root=0)
    prior_range = comm.bcast(prior_range, root=0)
   # fixed_param_indices = comm.bcast(fixed_param_indices, root=0)
    mean_prior = comm.bcast(mean_prior, root=0)
    std_prior = comm.bcast(std_prior, root=0)    
   # fixed_param_MLE = comm.bcast(fixed_param_MLE, root=0)
   # d_fixed = len(fixed_param_indices)
   # d_var = len(mean_prior)
    d_total = len(mean_prior)# d_fixed + d_var
    #var_param_indices = np.setdiff1d(np.arange( d_total), fixed_param_indices)

    #for T=0
    comm.barrier()
    #for T>0
    #print("starting SMC for T>0")
    for t in range(1,T):
        
        if rank == 0:
            print(f"t={t}", flush=True)
            valid_scores = population_scores[t-1,:]
            valid_scores = valid_scores[valid_scores!=None]
            valid_scores = valid_scores[~np.isnan(valid_scores)]
            epsilon_t = np.percentile(valid_scores, adaptive_percentile)
            epsilon_vector[t] = epsilon_t
            
        theta_t_minus = comm.bcast(theta_t_minus, root=0)
        epsilon_t = comm.bcast(epsilon_t, root=0)
        weights_old = comm.bcast(weights_old, root=0)
        kernel_cov_matrix = comm.bcast(kernel_cov_matrix, root=0)
        local_theta_star = []
        local_score =[]
        local_weights = []
        np.random.seed(42 + rank * 100 + t)
        for i in range(start,end):
           # print(i)
            theta_star, score, weight = iterative_theta_sample(weights_old, \
            theta_t_minus, kernel_cov_matrix, std_prior, mean_prior, epsilon_t,
            prior_list, prior_range)
            local_theta_star.append(theta_star)
            local_score.append(score)
            local_weights.append(weight)

        gathered_theta_star = comm.gather(local_theta_star, root=0)
        gathered_weights = comm.gather(local_weights, root=0)
        gathered_scores = comm.gather(local_score, root=0)

        if rank==0:
            scores=[]
            theta_stars = []
            weights = []
            for index, item in enumerate(gathered_scores):
                item_r = gathered_theta_star[index]
                item_w = gathered_weights[index]
                for index_2, item_2 in enumerate(item_r):
                    scores.append(item[index_2])
                    theta_stars.append(item_r[index_2])
                    weights.append(item_w[index_2])

            theta_t_minus = np.array(theta_stars)
            population_scores[t,:] = np.array(scores)
            weights_new = np.array(weights)
            particle_realizations[t,:,:] = theta_t_minus
            weight_normalization = np.sum(weights_new)
            if weight_normalization ==0 or not np.isfinite(weight_normalization):
                raise ValueError("Sum of new weights is either zero or non-finite-cannot normalize.")
            weights_old = np.exp(np.log(np.clip(weights_new, 1e-300, None))-np.log(weight_normalization))
            kernel_cov_matrix =  np.cov(theta_t_minus,rowvar=False)
            cov_scaling_constant = 1/np.max(std_prior*np.sqrt(np.diag(kernel_cov_matrix)))*np.log(2)
            kernel_cov_matrix = kernel_cov_matrix*(cov_scaling_constant)**2+ \
                              1e-10 * np.eye(d_var)
            print(f"Checkpointing after population {t}")
            save_checkpoint(particle_realizations, population_scores, t, epsilon_vector)



    if  rank==0 and not np.all(particle_realizations==0):
        np.save("accepted_realizations.npy", np.array(particle_realizations))
        np.save("accepted_scores.npy", np.array(population_scores))
        np.save("population_epsilons.npy", np.array(epsilon_vector))
