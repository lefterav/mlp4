import numpy as np
import sys
from scipy.stats import pearsonr
import pickle

from train import load_features, load_labels, clamp
from evaluation_measures import mean_absolute_error, root_mean_squared_error
import logging as log

def predict(model, features):
    if isinstance(model, list):
        labels_predicted = None
        for separate_model in model:
            labels_predicted_column = np.array([separate_model.predict(features)]).T
            if labels_predicted is None:
                labels_predicted = labels_predicted_column
            else:
                labels_predicted = np.concatenate((labels_predicted, labels_predicted_column), axis=1)
        
    else:
        labels_predicted = model.predict(features)
    return labels_predicted
    
def read_translation_lengths(translations_filename):
    translations_file = open(translations_filename)
    lengths = [len(t.strip().split(" ")) for t in translations_file]
    translations_file.close()
    return lengths
    
def evaluate(labels_predicted, labels_original, lengths, mode="--clamp_round"):
    if labels_predicted.ndim==2 and np.shape(labels_predicted)[1]==4:
        ter_predicted = []
        
        if mode=="--clamp_round":
            for (i,d,s,b), l in zip(labels_predicted, lengths):
                i = clamp(round(i, 0))
                d = clamp(round(d, 0), 0, l-1)
                s = clamp(round(s, 0), 0, l)
                b = clamp(round(b, 0), 0, l)
                ter_predicted.append(clamp((i+d+s+b)/(l+i-d), 0, 1))
        elif mode=="--clamp":
            for (i,d,s,b), l in zip(labels_predicted, lengths):
                i = clamp(i)
                d = clamp(d, 0, l-1)
                s = clamp(s, 0, l)
                b = clamp(b, 0, l)
                ter_predicted.append(clamp((i+d+s+b)/(l+i-d), 0, 1))
        elif mode=="--round":
            for (i,d,s,b), l in zip(labels_predicted, lengths):
                i = round(i, 0)
                d = round(d, 0)
                s = round(s, 0)
                b = round(b, 0)
                ter_predicted.append(clamp((i+d+s+b)/(l+i-d), 0, 1))
        elif mode=="--original":
            for (i,d,s,b), l in zip(labels_predicted, lengths):
                ter_predicted.append((i+d+s+b)*1.0/(l+i-d))
        
        
        #labels_predicted = [[clamp(round(l, 0)) for l in row] for row in labels_predicted]
        #for (pi,pd,ps,pb), (oi, od, os, ob), l in zip(labels_predicted, labels_original, lengths):
        #    print pi, oi, pd, od, ps, os, pb, ob
            
        #ter_predicted = [clamp((i+d+s+b)/(l+i-d), 0, 1) for (i,d,s,b), l in zip(labels_predicted, lengths)]
        #ter_original = [(i+d+s+b)/(l+i-d) for (i,d,s,b), l in zip(labels_original, lengths)]
        
    elif labels_predicted.ndim==2 and np.shape(labels_predicted)[1]==5:
        ter_predicted = [clamp(ter, 0, 1) for _, _, _, _, ter in labels_predicted]
    elif labels_predicted.ndim == 1:
        ter_predicted = labels_predicted
    
    pearson_r, pearson_p = pearsonr(ter_predicted, labels_original)
    mae = mean_absolute_error(ter_predicted, labels_original)
    rmse = root_mean_squared_error(ter_predicted, labels_original)
    pearson_r_inv, _ = pearsonr(labels_original, ter_predicted)
    return pearson_r, pearson_p, pearson_r_inv, mae, rmse

def evaluate_testfiles(model, feature_filename, labels_original_filename, translations_filename, mode="--clamp_round", scaled_test_features=None, scaler=None):
    if scaled_test_features is not None:
        features = scaled_test_features
    else:
        features = load_features(feature_filename)
        try:
            scaler = pickle.load(open("scaler.model"))
            features = scaler.transform(features)
            log.info("Sucessfully scaled features")
        except Exception as e:
            log.error("Could not scale features {}".format(e))
    labels_predicted = predict(model, features)
    labels_original = load_labels(labels_original_filename)
    translation_lengths = read_translation_lengths(translations_filename) 
    return evaluate(labels_predicted, labels_original, translation_lengths, mode)

if __name__ == "__main__":
    model_filename = sys.argv[1]
    test_feature_filename = sys.argv[2]
    test_translations_filename = sys.argv[3]
    test_labels_original_filename = sys.argv[4]
    
    try:
        mode = sys.argv[5] 
    except:
        mode = "--clamp_round"
    
    model = pickle.load(open(model_filename)) 
    print model_filename, evaluate_testfiles(model, test_feature_filename, test_labels_original_filename, test_translations_filename, mode)
    
    
    
    
