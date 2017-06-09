from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
import sys
from scipy.stats import pearsonr
import pickle
import logging as log
import time
from numpy.random import RandomState
from sklearn.metrics.pairwise import manhattan_distances

VERBOSE=False
JOBS=60

def clamp(n, minn=0, maxn=100):
    clamped = max(min(maxn, n), minn)
    if clamped == -0.0:
        clamped = abs(clamped)
    return clamped
        
        
def load_features(feature_filename):
    feature_vector_list = []
    feature_file = open(feature_filename)

    for featureline in feature_file:
        feature_vector = [float(f) for f in featureline.strip().split("\t")]
        feature_vector_list.append(feature_vector)
    
    feature_file.close()
    feature_array = np.array(feature_vector_list)
    return feature_array
    
    
def load_labels(label_filename):

    label_vector_list = []
    label_file = open(label_filename)
    
    for label_line in label_file:
        label_values = label_line.strip().split("\t")
        
        #check if this is a multi- or a single-value label
        if len(label_values) == 1:
            label_vector = float(label_values[0])
        else:
            label_vector = [float(l) for l in label_line.strip().split("\t")]
        label_vector_list.append(label_vector)
        
    label_array = np.array(label_vector_list)
    label_file.close()
    return label_array
    
def scorer_pearsonr(learner, features, original_labels, mode='edits_product_pearson'):
    predicted_labels = learner.predict(features)
    
    if predicted_labels.ndim==2 and mode=='edits_product_pearson':
        return pearsonr(predicted_labels[:,0], original_labels[:,0])[0] * \
               pearsonr(predicted_labels[:,1], original_labels[:,1])[0] * \
               pearsonr(predicted_labels[:,2], original_labels[:,2])[0] * \
               pearsonr(predicted_labels[:,3], original_labels[:,3])[0]
    elif predicted_labels.ndim==2 and mode=='hter_pearson':
        predicted_hter = []
        original_hter = []
        for (pi,pd,ps,pb), (oi,od,os,ob), featurevector in zip(predicted_labels, original_labels, features):

            l=featurevector[0]
            predicted_hter.append((pi+pd+ps+pb)/(l+pi-pd))
            original_hter.append((oi+od+os+ob)/(l+oi-od))
        return pearsonr(predicted_hter, original_hter)[0]
        
    elif predicted_labels.ndim==2 and mode=='hter_rounded_pearson':
        predicted_hter = []
        original_hter = []
        for (pi,pd,ps,pb), (oi,od,os,ob), featurevector in zip(predicted_labels, original_labels, features):
            pi = clamp(round(pi, 0))
            pd = clamp(round(pd, 0))
            ps = clamp(round(ps, 0))
            pb = clamp(round(pb, 0))
            l=featurevector[0]
            predicted_hter.append(clamp((pi+pd+ps+pb)/(l+pi-pd), 0, 1))
            original_hter.append(clamp((oi+od+os+ob)/(l+oi-od), 0, 1))
        return pearsonr(predicted_hter, original_hter)[0]
    else:
        return pearsonr(predicted_labels, original_labels)[0]


def scorer_mae(learner, features, original_labels):
    predicted_labels = learner.predict(features)
    vector = manhattan_distances(predicted_labels, original_labels)
    summation = np.sum(vector)
                     
    mae = -1.0* summation / original_labels.shape[0]
    
    return mae



def initialize_learner(learner_name):
    if learner_name == "MLP_3000_tanh":
        learner = MLPRegressor(hidden_layer_sizes=3000, activation='tanh', tol=0.000000001, alpha=0.01)
    elif learner_name == "MLP":
        learner = MLPRegressor()
    elif learner_name == "SVR":
        learner = SVR()
    elif learner_name == "SVR_opti":
        params =  {'C': [1, 10, 2], 'gamma': [0.001, 0.01, 2], 'epsilon': [0.1, 0.2, 2], 'kernel': ['rbf']}
        learner = GridSearchCV(SVR(), params, n_jobs=JOBS, cv=10, verbose=VERBOSE) 
    elif learner_name == "SVR_mae_opti":
        params =  {'C': [1, 10, 2], 'gamma': [0.001, 0.01, 2], 'epsilon': [0.1, 0.2, 2], 'kernel': ['rbf']}
        learner = GridSearchCV(SVR(), params, n_jobs=JOBS, cv=10, verbose=VERBOSE, scoring=scorer_mae) 
    elif learner_name == "SVR_pearson_opti":
        estimator = SVR()
        params =  {'C': [1, 10, 2], 'gamma': [0.001, 0.01, 2], 'epsilon': [0.1, 0.2, 2], 'kernel': ['rbf']}
        learner = GridSearchCV(estimator, params, n_jobs=JOBS, cv=10, verbose=VERBOSE, scoring=scorer_pearsonr) 
    elif learner_name == "SVR_best":
        learner = SVR(epsilon=0.1, C=10, gamma=0.001, kernel='rbf')
    elif learner_name == "SVR_organizers":
        learner = SVR(epsilon=0.0825, C=20, gamma=0.01, kernel='rbf')
    elif learner_name == "4xSVR_opti":
        params =  {'C': [1, 10, 2], 'gamma': [0.001, 0.01, 2], 'epsilon': [0.1, 0.2, 2], 'kernel': ['rbf']}
        learner = 4*[GridSearchCV(SVR(), params, n_jobs=JOBS, cv=10, verbose=VERBOSE, scoring=scorer_pearsonr)]        
    
    elif learner_name == "MLP_opti":
        estimator = MLPRegressor()
        params = {'activation': ['tanh', 'relu'], 'solver': [ 'adam'], 'alpha':[0.1, 0.01, 0.001], 'hidden_layer_sizes': [40, 45, 50, 57, 60, 62, 70, 75, 100, 110, 120, 140, 150, 160, 200, 300, 500, 1000], 'tol' : [0.000000001] }
        learner = GridSearchCV(estimator, params, n_jobs=JOBS, cv=10, verbose=VERBOSE, error_score=0)
    elif learner_name == "MLP_pearson_opti" or learner_name == "MLP1_pearson_opti":
        estimator = MLPRegressor()
        params = {'activation': ['tanh', 'relu'], 'solver': [ 'adam'], 'alpha':[0.1, 0.01, 0.001, 0.0001], 'hidden_layer_sizes': [40, 50, 57, 60, 70, 75, 100, 150, 200, 300, 500, 1000, 1100, 1500, 2000, 3000], 'tol' : [0.000000001] }
        learner = GridSearchCV(estimator, params, n_jobs=JOBS, cv=10, verbose=VERBOSE, error_score=0, scoring=scorer_pearsonr)
    elif learner_name == "MLP_best_de-en_edits":
        learner = MLPRegressor(alpha=0.01, activation='tanh', solver='adam', tol=1e-09, hidden_layer_sizes=300)
    elif learner_name == "MLP_best_en-de_edits":
        R = np.random.RandomState(int(time.time()))    
        print int(time.time())    
        learner = MLPRegressor(alpha=0.01, activation='tanh', solver='adam', tol=1e-09, hidden_layer_sizes=3000, random_state=R)
    elif learner_name == "MLPsgd_opti":
        estimator = MLPRegressor()
        params = {'activation': ['relu'], 'solver': [ 'sgd'], 'alpha':[0.01, 0.001, 0.0001, 0.00001], 'hidden_layer_sizes': [40, 42, 45, 50, 57, 60, 62, 70, 72, 75, 82, 100, 110, 120, 150], }
        learner = GridSearchCV(estimator, params, n_jobs=JOBS, cv=10, verbose=VERBOSE, error_score=0, scoring=scorer_pearsonr)
    return learner
    
def train(learner_name, features, labels):
    model = initialize_learner(learner_name)
    if isinstance(model, list):
        trained_model = []
        for column, separate_model in enumerate(model):
            separate_model.fit(features, labels[:, column])
            print "params:", column, separate_model.best_params_
            best_separate_model = separate_model.best_estimator_
            best_separate_model.fit(features, labels[:, column])
            trained_model.append(best_separate_model)
        return trained_model
        
    elif learner_name.endswith("_opti"):
        model.fit(features, labels)
        print "Params:", model.best_params_
        model = model.best_estimator_
    model.fit(features, labels)
    return model
    
def train_trainfiles(learner_name, feature_filename, label_filename):
    features = load_features(feature_filename)
    labels = load_labels(label_filename)
    model = train(learner_name, features, labels)
    return model

if __name__ == "__main__":
    learner_name = sys.argv[1]
    feature_filename = sys.argv[2]
    label_filename = sys.argv[3]
    
    model = train_trainfiles(learner_name, feature_filename, label_filename)
    model_filename = "{}.model".format(learner_name)
    pickle.dump(model, open(model_filename, 'w'))
    
    
        
    
    
   
    
    
    
