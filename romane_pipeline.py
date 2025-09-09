import pandas as pd 

from functions.generate_utils.preprocessing.processing_data import preprocessing_data
from sklearn.preprocessing import MinMaxScaler

from functions.generate_utils.models.VAE_MMD_VAE_autoencoder.processing_inputs_data import preprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam





mrna_data = pd.read_csv('data/BRCA_mRNA.csv', index_col=0)
cnv_data = pd.read_csv('data/BRCA_CNV.csv', index_col=0)
methyl_data = pd.read_csv('data/BRCA_Methy.csv', index_col=0)
mirna_data = pd.read_csv('data/BRCA_miRNA.csv', index_col=0)
clinical_data = pd.read_csv('data/Clinical_Rec.csv', index_col=0)



# filter the data to include same patients 
clinical_data_filtered, mrna_data_filtered, cnv_data_filtererd, methyl_data_filtered, mirna_data_filtered = preprocessing_data(mrna_data, cnv_data, methyl_data, mirna_data, clinical_data)



# 1) Data preprocessing (normalization (min-max), same patients order, remove genes with missing data)

common_cols = sorted(set(mrna_data_filtered.columns) & set(cnv_data_filtererd.columns) & set(methyl_data_filtered.columns) & set(mirna_data_filtered.columns))
mrna_data_filtered = preprocess(mrna_data_filtered, common_cols)
cnv_data_filtererd = preprocess(cnv_data_filtererd, common_cols)
methyl_data_filtered = preprocess(methyl_data_filtered, common_cols)

combined_df = pd.concat([mrna_data_filtered, cnv_data_filtererd, methyl_data_filtered], axis=0)




# create a classification column

# 1825

clinical_data_filtered.loc['survival_classification'] = (clinical_data_filtered.loc['days'] > 1825).astype(int)
print(clinical_data_filtered.loc['survival_classification'].value_counts())

# an MLP followed by a Cox-PH model, 




# 2) dimension reduction VAE/MMD-VAE
# Train model VAE or MMD-VAE with multiomics datasets (unsupervised setting)-> encoder and decoder 
# encoder - compresses high dim data into a low dim latent space (related to and helps manage the imbalanced size between the number of samples and the number of features)
# decoder - reconstruct the original input 
# once trained - encoder outputs is set of latent features (LFs)-> most important features learned 
















# Example usage for different modalities
# Let's assume you have clinical data, genomic data, and image features
# clinical_encoder = MLPEncoder(input_dim=50, hidden_dim=128, output_dim=64)
# genomic_encoder = MLPEncoder(input_dim=1000, hidden_dim=256, output_dim=64)
# imaging_encoder = MLPEncoder(input_dim=2048, hidden_dim=512, output_dim=64)









# 3) survival analysis 
# identify clinically relevant LFs (univasrriate univariate Cox Proportional Hazards (Cox-PH) regression) on each of the LFs
# identify the LFs that have association with patient survival

# infer survival subgroups: use a clustering algorithm like k-means to identify clinically relevant LFs to group 
# evaluate perf: statistical measures to evaluate survival prediction : C-index, log rank p value, Brier score



# identify potential biomarker 
#  trying to map the latent features (LFs) back to the original omics data to find potential biomarkers
# correlation analysis (establish the link between clincially relevant LFs and the original input features - genes, CpG islands)
# identify the genes/ proteins most strongly associated with the LFs 
# filtering - filter the features with zero or insignificant values from the corr analayis 