import pandas as pd 
from functions.generate_utils.processing_data import preprocessing_data
import numpy as np
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.linear_model import CoxnetSurvivalAnalysis


mrna_data = pd.read_csv('data/BRCA_mRNA.csv', index_col=0)
cnv_data = pd.read_csv('data/BRCA_CNV.csv', index_col=0)
methyl_data = pd.read_csv('data/BRCA_Methy.csv', index_col=0)
mirna_data = pd.read_csv('data/BRCA_miRNA.csv', index_col=0)
clinical_data = pd.read_csv('data/Clinical_Rec.csv', index_col=0)



clinical_data_filtered, mrna_data_filtered, cnv_data_filtererd, methyl_data_filtered, mirna_data_filtered, median_survival = preprocessing_data(mrna_data, cnv_data, methyl_data, mirna_data, clinical_data)

# mrna_transpose = mrna_data.T

X = mrna_data_filtered.T



time = clinical_data_filtered.loc['days']
event = clinical_data_filtered.loc['status']


Y = Surv.from_arrays(event=event.astype(bool), time=time.astype(float))



#train test split - 75/25
X_train, X_test, y_train, y_test, event_train, event_test, time_train, time_test = train_test_split(
X, Y, event, time, test_size=0.25, random_state=42, stratify=event)


model = CoxnetSurvivalAnalysis(l1_ratio=0.9, alphas=None)

# Fit the model to the training data
model.fit(X_train, y_train)





# array_from_mrna = mrna_transpose.values


# events_list = list(clinical_data_filtered.loc['status'])
# times_list = list(clinical_data_filtered.loc['days'])


# y_train = np.column_stack((events_list, times_list))


# X_train, X_test, y_train, y_test = train_test_split(
#     data_x, data_y, test_size=0.25, random_state=42
# )


# print(y_train)



# # # y: structured array with fields ('event', 'time')
# # # e.g., y = Surv.from_arrays(event=y_event.astype(bool), time=y_time.astype(float))


# coxnet = CoxnetSurvivalAnalysis(l1_ratio=0.5).fit(array_from_mrna, y_train)

# # # 1) Log-risk score (linear predictor). Higher -> riskier.
# risk_score = coxnet.predict(array_from_mrna)           # shape (n_test,)

# # 2) Survival curves S(t|x) as step functions
# s_funcs = coxnet.predict_survival_function(X_test)  # list of StepFunction

# # Probability of surviving >= T years for each sample:
# T = 5.0
# p_survive_T = np.array([sf(T) for sf in s_funcs])

# # Predicted median survival time for each sample:
# def median_survival_from_sf(sf):
#     # find first time t where S(t) <= 0.5
#     t = sf.x                      # times
#     s = sf.y                      # survival values
#     idx = np.where(s <= 0.5)[0]
#     return float(t[idx[0]]) if len(idx) else np.inf  # inf = median not reached

# median_pred = np.array([median_survival_from_sf(sf) for sf in s_funcs])

# # 3) Hazard ratios between two patients i and j:
# i, j = 0, 1
# hazard_ratio_ij = float(np.exp(risk_score[i] - risk_score[j]))


