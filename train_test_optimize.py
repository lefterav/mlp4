from train import train_trainfiles
from test import evaluate_testfiles
import sys
import pickle

if __name__ == "__main__":
    learner_name = sys.argv[1]
    feature_filename = sys.argv[2]
    label_filename = sys.argv[3]
    
    test_feature_filename = sys.argv[4]
    test_translations_filename = sys.argv[5]
    test_labels_original_filename = sys.argv[6]
    
    print learner_name
    
    scores = (0,0,0,0)
    while scores[0] < 0.387:
        model = train_trainfiles(learner_name, feature_filename, label_filename)

    
        model_filename = "{}.model".format(learner_name)
        pickle.dump(model, open(model_filename, 'w'))
        
        scores = evaluate_testfiles(model, test_feature_filename, test_labels_original_filename, test_translations_filename)
        print "", "\t".join([str(s) for s in scores])
