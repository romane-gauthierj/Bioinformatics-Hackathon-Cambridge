import pandas as pd 

from functions.generate_utils.processing_data import preprocessing_data
from functions.generate_utils.survival_model import run_CoxnetSurvivalAnalysis



mrna_data = pd.read_csv('data/BRCA_mRNA.csv', index_col=0)
cnv_data = pd.read_csv('data/BRCA_CNV.csv', index_col=0)
methyl_data = pd.read_csv('data/BRCA_Methy.csv', index_col=0)
mirna_data = pd.read_csv('data/BRCA_miRNA.csv', index_col=0)
clinical_data = pd.read_csv('data/Clinical_Rec.csv', index_col=0)



# process the data
clinical_data_filtered, mrna_data_filtered, cnv_data_filtererd, methyl_data_filtered, mirna_data_filtered = preprocessing_data(mrna_data, cnv_data, methyl_data, mirna_data, clinical_data)



# perform survival analysis
risk_scores, c_index, perf_score = run_CoxnetSurvivalAnalysis(0.9, mirna_data_filtered, clinical_data_filtered)





# TODO 
# add cross validation part; add the scaler 