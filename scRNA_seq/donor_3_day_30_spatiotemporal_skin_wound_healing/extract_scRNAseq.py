# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:20:43 2025

@author: alira
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
from scipy.sparse import save_npz
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
from scipy.sparse import csr_matrix, issparse
import scanpy as sc
import anndata
import datetime
import scrublet as scr
#import diffxpy.api as de
import singler

# Load the TSV file
# Load sparse count matrix
matrix = mmread("matrix.mtx").tocsc()  # Convert to Compressed Sparse Column format
print("Matrix shape:", matrix.shape)  # Should be (genes, cells)
genes = pd.read_csv("features.tsv", sep="\t", header=None)[1]  # Extract gene names
#cells = pd.read_csv("barcodes.tsv", sep="\t", header=None)[0]  # Extract cell barcodes
matrix = matrix.T #express cells as rows, genes as numbers
mito_genes = [gene for gene in range(len(genes)) if 'MT-' in genes[gene]]
# mito_genes now holds the indices of genes that are mitochondrial


"""
Apply first layer of filtering
"""
# Assuming 'matrix' is the expression matrix (cells x genes)
# Sum the expression of mitochondrial genes in each cell
mito_genes_expression = np.sum(matrix[:,mito_genes], axis=1)
# Now we can compute the percentage of mitochondrial gene expression per cell
total_counts_expression_per_cell = np.sum(matrix, axis=1)  # Total counts per cell
mito_percentage_per_cell = mito_genes_expression / total_counts_expression_per_cell  # Percentage of mitochondrial gene expression

min_gene_number = 500  #this should be 200.
max_gene_number = 8000
min_expression_count=500
umi_counts_per_cell = np.sum(matrix > 0, axis=1)  # Count of non-zero values (i.e. number genes) per cell
total_counts_per_cell = np.sum(matrix, axis=1) #total count per cell
filtered_cells = (umi_counts_per_cell > min_gene_number) & (umi_counts_per_cell < max_gene_number) &\
    (total_counts_per_cell > min_expression_count) & (mito_percentage_per_cell < 0.2)
filtered_matrix = matrix[np.where(filtered_cells==True)[0],:]



#Apply scrublet filtering to remove double cell detection
scrub = scr.Scrublet(filtered_matrix)
doublet_scores, predicted_doublets = scrub.scrub_doublets()

# Set the threshold for doublet detection (mean + 2 * absolute deviation)
#threshold = np.mean(doublet_scores) + 2 * np.std(doublet_scores)
#predicted_doublets = doublet_scores > threshold  # Cells above threshold are predicted doublets

# Step 4: Exclude predicted doublets
filtered_matrix_2 = filtered_matrix[np.where(predicted_doublets == 0)[0],:]  # Only include cells that are not doublets
filtered_matrix_2 = filtered_matrix_2.toarray()
#Now final_filtered_matrix contains only the cells that passed the filtering criteria and doublet removal.


"""
Optional: apply second layer of filtering
"""
#filtered_matrix_3 = np.load("filtered_matrix_3.npy")

# Check if it's already sparse
if not issparse(filtered_matrix_2):
    filtered_matrix_2 = csr_matrix(filtered_matrix_2)  # Convert to CSR format



umi_counts_per_cell = np.sum(filtered_matrix_2 > 0, axis=1)  # Count of non-zero values (i.e. number genes) per cell
#umi_counts_per_cell = filtered_matrix_3.count_nonzero(axis=1)
total_counts_per_cell = np.sum(filtered_matrix_2, axis=1) #total count per cell
filtered_cells = (umi_counts_per_cell > min_gene_number) & \
    (umi_counts_per_cell < max_gene_number) &\
    (total_counts_per_cell > min_expression_count)
filtered_matrix_3 = filtered_matrix_2[np.where(filtered_cells==True)[0],:]

#Now filter out genes that are expressed in less than 10 cells
cells_per_gene = np.array(filtered_matrix_3.sum(axis=0)).flatten()  # Convert to 1D array
# Apply filtering
final_filtered_matrix = filtered_matrix_3[:, np.where(cells_per_gene > 10)[0]]
save_npz('final_filtered_matrix.npz', final_filtered_matrix)
# Step 1: Identify genes that survive the filter
genes_kept = genes[np.where(cells_per_gene > 10)[0]]  # Indices of genes that survive first filtering


# Save the final genes list
genes_kept.to_csv('final_genes_kept.txt', index=False, header=False)


#Processessing: normalization and log-transform and remove genes with low variance
print("Min value:", final_filtered_matrix.min())
print("Max value:", final_filtered_matrix.data.max())
print("Mean value:", final_filtered_matrix.mean())


"""Apply the normalization and log-transform
matrix_norm = normalize(final_filtered_matrix, axis=1, norm="l1") * 1e4  # Normalize to 10,000 total reads per cell

#Log-transform
#matrix_norm_log = np.log(1+matrix)
matrix_norm_log = matrix_norm
matrix_norm_log.data = np.log1p(matrix_norm_log.data)  # Apply log to nonzero values only
#gene_variances = matrix_norm_log.var(axis=0)
# Compute mean per gene (axis=0 means across cells)
gene_means = matrix_norm_log.mean(axis=0).A1  # Convert sparse result to 1D NumPy array
"""

"""
Use sctransform from scanpy to do the transformation (exported to google collab)
"""
#final_filtered_matrix = np.load("final_filtered_matrix.npz",allow_pickle=True)
#data = final_filtered_matrix['data']
#indices = final_filtered_matrix['indices']
#indptr = final_filtered_matrix['indptr']
#shape = final_filtered_matrix['shape']
#final_filtered_matrix = sparse.csr_matrix((data, indices, indptr), shape=shape)
adata = sc.AnnData(final_filtered_matrix)
#adata.X = adata.X.toarray()



# Update var_names with gene names
adata.var_names = genes_kept.astype(str).values  # Force string type
adata.var_names_make_unique()  # Ensure unique names
# To keep the original indices as well
adata.var['original_index'] = adata.var.index
adata.layers["counts"] = adata.X.copy()



# Proceed with normalization and other preprocessing steps
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)




sc.pp.highly_variable_genes(adata, n_top_genes=4000, flavor='seurat')




if "highly_variable" not in adata.var.columns:
    raise ValueError("Highly variable genes were not computed correctly!")


#perform dimensionality reduction and clustering
adata_hvg = adata[:, adata.var.highly_variable].copy()
sc.tl.pca(adata_hvg, n_comps=50)
highly_variable_genes = adata.var.index[adata.var['highly_variable']]
sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=50)
sc.tl.umap(adata_hvg)
sc.tl.leiden(adata_hvg, flavor='igraph' , resolution=0.8)

#find differentially_expressed_genes
#sc.tl.rank_genes_groups(adata_hvg, groupby='leiden', method='wilcoxon')
# Perform differential expression analysis per cluster
# adata_hvg.X = adata_hvg.X + 1e-7 #prevent log(o) during de differential testing
# results = {}
# for cluster in adata_hvg.obs["leiden"].unique():
#     try:
#         # Create comparison vector
#         subset = adata_hvg.copy()
#         subset.obs["comparison"] = (subset.obs["leiden"] == cluster).astype(int)

#         # More stringent gene filtering
#         gene_means = np.mean(subset.X, axis=0)
#         gene_vars = np.var(subset.X, axis=0)

#         # Filter genes with low mean AND low variance
#         keep_genes = (gene_means > 0.1) & (gene_vars > 0.01)
#         subset = subset[:, keep_genes]

#         # Additional checks
#         if subset.shape[1] < 50:  # Require minimum genes
#             print(f"Cluster {cluster} skipped - only {subset.shape[1]} genes pass filters")
#             continue

#         if np.any(~np.isfinite(subset.X)):
#             print(f"Cluster {cluster} has non-finite values - skipping")
#             continue

#         # Add stronger regularization
#         test = de.test.wald(
#             data=subset,
#             formula_loc="~ comparison",
#             factor_loc_totest="comparison",
#             noise_model="nb",  # Explicitly specify negative binomial
#             training_strategy="DEFAULT",  # Uses more stable defaults
#             init_a=1.0,  # Stronger initialization
#             init_b=1.0
#         )

#         # Extract results with additional checks
#         summary = test.summary()
#         valid_results = summary[(summary["p_val_adj"] < 0.05) &
#                               (np.isfinite(summary["avg_log2FC"]))]
#         top_genes = valid_results.nlargest(20, "avg_log2FC")["gene"].tolist()
#         results[cluster] = set(top_genes)

#     except Exception as e:
#         print(f"Failed for cluster {cluster}: {str(e)}")
#         continue

#     # Extract the top 20 marker genes
#     top_genes = test.summary().query("p_val_adj < 0.05").nlargest(20, "avg_log2FC")["gene"].tolist()
#     results[cluster] = set(top_genes)

sc.tl.rank_genes_groups(
    adata_hvg,
    groupby='leiden',
    method='wilcoxon',
    key_added='de_results',  # New key
    pts=True
)

marker_df = pd.read_excel('The top 50 marker genes of the 27 cell clusters identified by scRNA-seq of human skin and acute wounds.xlsx')

# Create cluster-to-gene dictionary
study_markers = {}
for cluster, group in marker_df.groupby('Cluster'):
    study_markers[cluster] = set(group['gene'].str.upper())  # Case-insensitive matching


min_genes = 2  # Require ≥2 marker genes to match

for cluster in adata_hvg.obs['leiden'].unique():
    # Get this cluster's top DEGs (uppercase for matching)
    cluster_genes = set(
        sc.get.rank_genes_groups_df(
            adata_hvg,
            group=str(cluster),
            key='de_results'
        )['names'].str.upper().head(50)  # Check top 50 genes
    )

    # Find best-matching cell type from study
    best_match = None
    best_score = 0
    for cell_type, markers in study_markers.items():
        overlap = len(cluster_genes & markers)
        if overlap > best_score:
            best_score = overlap
            best_match = cell_type

    # Assign label if sufficient overlap
    if best_score >= min_genes:
        adata_hvg.obs.loc[adata_hvg.obs['leiden'] == cluster, 'lineage'] = best_match
    else:
        adata_hvg.obs.loc[adata_hvg.obs['leiden'] == cluster, 'lineage'] = f"Unknown_{cluster}"

# Propagate to full dataset
adata.obs['lineage'] = adata_hvg.obs['lineage'].reindex(adata.obs.index)




sc.pl.umap(adata_hvg, color='lineage')


#Get cell count of each cell type
#lineage_counts = adata_hvg.obs['lineage'].value_counts()

# Get lineage counts
lineage_counts = adata_hvg.obs['lineage'].value_counts().sort_values(ascending=False)

# Create the stacked histogram
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(lineage_counts.index, lineage_counts.values, edgecolor="black")
ax.bar_label(bars, fmt='%d', label_type='edge')

# Customize the plot
ax.set_xlabel('Lineage')
ax.set_ylabel('Cell Count')
#ax.set_title('Cell Count per Lineage (Stacked)')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# # Add value labels on top of each bar
# for i, v in enumerate(lineage_counts.values):
#     ax.text(i, v, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()


#For NPJ submission combined graph

# Panel (a): UMAP
fig = plt.figure(figsize=(10, 12))  
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.35)

# --- Subplot (a) ---
ax1 = fig.add_subplot(gs[0])
sc.pl.umap(
    adata_hvg, 
    color='lineage', 
    ax=ax1, 
    show=False,
    legend_loc='on data'  # or "right margin"
)
ax1.text(-0.05, 1.05, "(a)", transform=ax1.transAxes, fontsize=16, fontweight="bold")

# --- Subplot (b) ---
ax2 = fig.add_subplot(gs[1])

# Lineage counts
lineage_counts = adata_hvg.obs['lineage'].value_counts().sort_values(ascending=False)

bars = ax2.bar(lineage_counts.index, lineage_counts.values, edgecolor="black")
ax2.bar_label(bars, fmt='%d', label_type='edge')

ax2.set_xlabel('Lineage')
ax2.set_ylabel('Cell Count')
plt.xticks(rotation=45, ha='right')

ax2.text(-0.05, 1.05, "(b)", transform=ax2.transAxes, fontsize=16, fontweight="bold")

plt.tight_layout()

# Save final figure (NO caption, NO "Figure 1" text)
plt.savefig("Figure1.png", dpi=600, bbox_inches="tight")
plt.savefig("Figure1.pdf", dpi=600, bbox_inches="tight")

plt.show()




"""
Construct adata that has mixture of types and subtypes for spatial analysis
"""
#adata1 = sc.read_h5ad('gscRNA_data_before_log_norm.h5ad')
#adata.obs['cell type'] = adata1.obs['lineage'].copy()  # Ensure 'lineage' is in adata1.obs

adata.obs["cell type"] = adata_hvg.obs["lineage"].reindex(adata.obs.index)
# Save the updated adata2 to a new file
adata.write('global_adata_with_celltypes.h5ad')
adata_hvg.write('global_processed_adata_hvg_with_celltypes.h5ad')
