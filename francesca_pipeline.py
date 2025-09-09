import pandas as pd
import numpy as np
from functions.generate_utils.processing_data import preprocessing_data
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from functions.generate_utils.survival_model import run_CoxnetSurvivalAnalysis


mrna_data = pd.read_csv('data/BRCA_mRNA.csv', index_col=0)
cnv_data = pd.read_csv('data/BRCA_CNV.csv', index_col=0)
methyl_data = pd.read_csv('data/BRCA_Methy.csv', index_col=0)
mirna_data = pd.read_csv('data/BRCA_miRNA.csv', index_col=0)
clinical_data = pd.read_csv('data/Clinical_Rec.csv', index_col=0)


clinical_data_filtered, mrna_data_filtered, cnv_data_filtererd, methyl_data_filtered, mirna_data_filtered = preprocessing_data(mrna_data, cnv_data, methyl_data, mirna_data, clinical_data)



# print(mirna_data_filtered)
#print(cnv_data_filtererd)
print(clinical_data_filtered)



# #defining x and y - x=biomarkers, y=survival
X = mirna_data_filtered.T
time = clinical_data_filtered.loc['days']
event = clinical_data_filtered.loc['status']
Y = Surv.from_arrays(event=event.astype(bool), time=time.astype(float))
len(Y)

# #train test split - 75/25
X_train, X_test, y_train, y_test, event_train, event_test, time_train, time_test = train_test_split(
    X, Y, event, time, test_size=0.25, random_state=42, stratify=event
)

# # Scale within train only
scaler = StandardScaler().fit(X_train)
X_train_z = scaler.transform(X_train)
X_test_z  = scaler.transform(X_test)

# # Fit Cox elastic net with defaults
coxnet = CoxnetSurvivalAnalysis(l1_ratio=0.9, alphas=None)  # l1_ratio=0.5, alpha path chosen automatically
coxnet.fit(X_train_z, y_train)

# # Risk scores and test C-index
risk_test = coxnet.predict(X_test_z)
cindex = concordance_index_censored(y_test["event"], y_test["time"], risk_test)[0]
print("Test C-index:", round(cindex, 3))


#model score
model_score = coxnet.score(X_test, y_test)
model_score

risk_scores, c_index, perf_score = run_CoxnetSurvivalAnalysis(0.9, mrna_data_filtered, clinical_data_filtered)
c_index