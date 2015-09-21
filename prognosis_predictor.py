import pandas as pd
import numpy as np
import json
from collections import Counter
import random
from math import ceil
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

ICD9_DGNS_CD = ['ICD9_DGNS_CD_1','ICD9_DGNS_CD_2','ICD9_DGNS_CD_3','ICD9_DGNS_CD_4',
                'ICD9_DGNS_CD_5','ICD9_DGNS_CD_6','ICD9_DGNS_CD_7','ICD9_DGNS_CD_8',
                'ICD9_DGNS_CD_9','ICD9_DGNS_CD_10']
                
def main():
    inpatient_claims = pd.read_csv('DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv')
    outpatient_claims = pd.read_csv('DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv')
    cols = ICD9_DGNS_CD + ['DESYNPUF_ID', 'CLM_FROM_DT']
    inpatient_claims = inpatient_claims[cols]
    outpatient_claims = outpatient_claims[cols]
    claims = pd.concat([inpatient_claims, outpatient_claims])
    datapoints = get_claims_attribute(claims, 2008, 2009, '5855', '5856')
    #claims = Counter()
    #for patient in datapoints:
    #    for c in datapoints[patient]['ccs_codes']:
    #        claims[c] += 1
    #print(claims)
    add_summary(datapoints, 2008)
    X,Y = matrix_format(datapoints)
    training_samples = set(random.sample(range(len(X)), ceil(0.7 * len(X))))
    Xtrain = []
    Ytrain = []
    Xtest = []
    Ytest = []
    for i in range(len(X)):
        if i in training_samples:
            Xtrain.append(X[i])
            Ytrain.append(Y[i])
        else:
            Xtest.append(X[i])
            Ytest.append(Y[i])
    print('logistic regression')
    lr = LogisticRegression()
    lr.fit(Xtrain, Ytrain)
    C = lr.predict(Xtest)
    error = 0.0
    for i,b in enumerate(C):
        if Ytest[i] == b:
            pass
        else:
            error += 1
    print(error / len(C))
    print('random forest')
    rf = RandomForestClassifier()
    rf.fit(Xtrain, Ytrain)
    error = 0.0
    C = rf.predict(Xtest)
    error = 0.0
    for i,b in enumerate(C):
        if Ytest[i] == b:
            pass
        else:
            error += 1
    print(error / len(C))
    print('svm')
    sv = svm.SVC()
    sv.fit(Xtrain, Ytrain)
    error = 0.0
    C = sv.predict(Xtest)
    for i,b in enumerate(C):
        if Ytest[i] == b:
            pass
        else:
            error += 1
    print(error / len(C))
    
def matrix_format(data_points):
    # age / ismale / diabetes / nutrition deficiences / hypertension / infectious disease /
    # vascular issues / smoking / uti
    dim = 9
    diabetes = set(['49', '50'])
    nut_defic = set(['59', '55', '53', '58', '52', '51'])
    hypertension = set(['98', '99'])
    inf_disease = set(['2', '10', '122'])
    vasc = set(['108', '106', '117', '114', '118', '96'])
    smoking = set(['127', '133', '138',])
    uti = set(['159'])
    comorbids = [diabetes, nut_defic, hypertension, inf_disease, vasc, smoking, uti]
    X = []
    Y = []
    for patient in data_points:
        patient_dict = data_points[patient]
        is_male_binary = int(patient_dict['male'])
        vec = [patient_dict['age'], is_male_binary]
        for cm in comorbids:
            if len(patient_dict['ccs_codes'] & cm) >= 1:
                vec.append(1)
            else:
                vec.append(0)
        X.append(vec)
        Y.append(int(patient_dict['final_stage']))
    return X,Y
        
    
def add_summary(data_points, year):
    summary = pd.read_csv('DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv')
    patients = set([patient for patient in data_points])
    for _, patient in summary.iterrows():
        if patient['DESYNPUF_ID'] in patients:
            age = year - (patient['BENE_BIRTH_DT'] // 10000)
            if patient['BENE_SEX_IDENT_CD'] == 1:
                is_male = True
            else:
                is_male = False
            data_points[patient['DESYNPUF_ID']]['age'] = age
            data_points[patient['DESYNPUF_ID']]['male'] = True
    
                
def get_claims_attribute(claims, start_year, end_year, icd9_code, future_icd9_code):
    # filter everything that is not correct year
    claims_start = filter_claims_year(claims, start_year)
    # filter to get all claims that mention icd9_code
    claims_filtered = claims_start[get_icd9_filter(claims_start, icd9_code)]
    # remove those that also mention code to exclude
    claims_filtered = claims_filtered[~get_icd9_filter(claims_filtered, future_icd9_code)]
    # get all patient ids
    patients = set(claims_filtered['DESYNPUF_ID'])
    # map each patient to set of ccs codes and whether they progressed to final stage
    datapoints = {}
    for patient in patients:
        datapoints[patient] = {'ccs_codes': set(), 'final_stage': False}
    # get all claims in start_year by people with this id
    total_claims_start = filter_claims_patients(claims_start, patients)
    # build json to convert to ccs codes
    with open('icd2singleccs.json') as f:
        json_text = f.read()
    ccs_json = json.loads(json_text)
    for _, claim in total_claims_start.iterrows():
        # add to each patient relevant ccs codes
        patient = claim['DESYNPUF_ID']
        patient_ccs = set()
        for cd in ICD9_DGNS_CD:
            if not pd.isnull(claim[cd]):
                ccs = ccs_json.get(claim[cd], None)
                if ccs is not None:
                    datapoints[patient]['ccs_codes'].add(ccs)
    # now get all patients who progressed to final stage from our set
    claims_end = filter_claims_year(claims, end_year)
    claims_end_filtered = claims_end[get_icd9_filter(claims_end, future_icd9_code)]
    claims_end_filtered = filter_claims_patients(claims_end_filtered, patients)
    progressed_patients = set(claims_end_filtered['DESYNPUF_ID'])
    for patient in progressed_patients:
        datapoints[patient]['final_stage'] = True
    return datapoints

def filter_claims_year(claims, year):
    get_year = lambda y: y.astype(np.int64) // 10000
    claims_year = claims['CLM_FROM_DT'].apply(get_year)
    return claims[claims_year == year]
    
def filter_claims_patients(claims, patients):
    return claims[claims['DESYNPUF_ID'].apply(lambda x: x in patients)]

def get_icd9_filter(claims, icd9_code):
    flt = claims[ICD9_DGNS_CD[0]] == icd9_code
    for cd in ICD9_DGNS_CD[1:]:
        flt |= claims[cd] == icd9_code
    return flt
    
if __name__ == '__main__':
    main()
    
    
    
        