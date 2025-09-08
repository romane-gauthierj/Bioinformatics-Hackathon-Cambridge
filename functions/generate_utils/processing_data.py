import pandas as pd 
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter






# keep the BRCA clinical data 
# check the gender distribution
# ensure consistency in the samples names
# look at the survival time distribution -> decide survival 
# in other datasets only keep the samples ID of the clinical data -> integration 
# check final nb of samples 


def preprocessing_data(mrna_data, cnv_data, methyl_data, mirna_data, clinical_data):

    # preprocessing
    mrna_data.index.name = "gene_name"
    cnv_data.index.name = "gene_name"
    methyl_data.index.name = "gene_name"
    mirna_data.index.name = "gene_name"


    #preprocessing clinical data
    clinical_data.index = clinical_data.index.str.replace("-", ".", regex=False)
    clinical_data_filtered = clinical_data[clinical_data['label']=='BRCA']
    clinical_data_filtered = clinical_data_filtered[clinical_data_filtered['gender']=='FEMALE']


    clinical_data_filtered = clinical_data_filtered.transpose()

    cols_to_drop = clinical_data_filtered.loc[["days", "status"]].isna().any(axis=0)

    clinical_data_filtered = clinical_data_filtered.loc[:, ~cols_to_drop]
    patients_ids = list(clinical_data_filtered.columns)

    valid_ids_mran = [col for col in patients_ids if col in mrna_data.columns]
    mrna_data_filtered = mrna_data[valid_ids_mran]

    valid_ids_cnv = [col for col in patients_ids if col in cnv_data.columns]
    cnv_data_filtererd = cnv_data[valid_ids_cnv]

    valid_ids_methyl = [col for col in patients_ids if col in methyl_data.columns]
    methyl_data_filtered = methyl_data[valid_ids_methyl]

    valid_ids_mirna = [col for col in patients_ids if col in mirna_data.columns]
    mirna_data_filtered = mirna_data[valid_ids_mirna]
    final_patients_ids = list(set(valid_ids_mran) & set(valid_ids_cnv) & set(valid_ids_methyl) & set(valid_ids_mirna))

    clinical_data_filtered = clinical_data_filtered[final_patients_ids]

    return clinical_data_filtered, mrna_data_filtered, cnv_data_filtererd, methyl_data_filtered, mirna_data_filtered



