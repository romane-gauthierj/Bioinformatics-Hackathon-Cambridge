import pandas as pd
import numpy as np
from functions.generate_utils.processing_data import preprocessing_data


mrna_data = pd.read_csv('data/BRCA_mRNA.csv', index_col=0)
cnv_data = pd.read_csv('data/BRCA_CNV.csv', index_col=0)
methyl_data = pd.read_csv('data/BRCA_Methy.csv', index_col=0)
mirna_data = pd.read_csv('data/BRCA_miRNA.csv', index_col=0)
clinical_data = pd.read_csv('data/Clinical_Rec.csv', index_col=0)



mrna_data_filtered, cnv_data_filtererd, methyl_data_filtered, mirna_data_filtered, median_survival = preprocessing_data(mrna_data, cnv_data, methyl_data, mirna_data, clinical_data)


print(median_survival)