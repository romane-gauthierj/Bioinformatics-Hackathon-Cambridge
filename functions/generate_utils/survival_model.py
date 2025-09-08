
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored


def run_CoxnetSurvivalAnalysis(l1_ratio, data_interest, clinical_data_filtered):
    #l1_ratio = 0.9
    X = data_interest.T

    time = clinical_data_filtered.loc['days']
    event = clinical_data_filtered.loc['status']

    Y = Surv.from_arrays(event=event.astype(bool), time=time.astype(float))

    #train test split - 75/25
    X_train, X_test, y_train, y_test, event_train, event_test, time_train, time_test = train_test_split(
    X, Y, event, time, test_size=0.25, random_state=42, stratify=event)


    # Model Training
    model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=None)

    # Fit the model to the training data
    model.fit(X_train, y_train)


    # Prediction and Evaluation
    risk_scores = model.predict(X_test)
    c_index, _, _, _, _ = concordance_index_censored(y_test['event'], y_test['time'], risk_scores)

    perf_score = model.score(X_test, y_test) # evaluate model's performance (C-index), score close to 1 = perfect prediction

    return risk_scores, c_index, perf_score


