# Genetic_analysis
Includes codes for single-cell RNA sequencing analysis, visium spatial transcriptomics and cell2location analysis, cellphonedb, and the diffusion model, as well the ABC and gradient descent schemes for parameter calibration.
The order is as follows:
  1. scRNA-seq analysis to obtain annotation of cells by cell type----->annotated scRNAseq data
  2. CellPhoneDB (or any other package for infererring interaction strengths) on annotated scRNA-seq data--------->interaction strengths
  3. Visium preanalysis code on spatial transcriptomics (ST) data-------->positions of spots, neighbourhood info, mesh info, processed ST data
  4. Cell2Location Analysis using the annoated scRNA-seq data and the processed ST data--------> estimates (q05,mean, q95, std) of cell number per type per spot
  5. MC (ABC rejection scheme) run and analysis ------->SMC starting particles.
  6. SMC with SMC starting particles-----------> final population particles
  7. (optional) SMC analysis
  8. Gradient-based optimization using starting seeds from the final SMC particles (Newton or ADAM) --------> final parameters
  9. Posthoc-analysis: Confidence interval (CI) estimation and analysis, comparison between samples....etc.
