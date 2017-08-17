from train import train_trainfiles, scale_and_train_files
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
    try:
        if sys.argv[7] == "--scale":
            scale = True
    except:
        scale = False
    
    print learner_name
    
    test_features = None
    if scale:    
        model, test_features = scale_and_train_files(learner_name, feature_filename, label_filename, test_feature_filename)
    else:
        model = train_trainfiles(learner_name, feature_filename, label_filename)

    model_filename = "{}.model".format(learner_name)
    pickle.dump(model, open(model_filename, 'w'))
    
    scores = evaluate_testfiles(model, test_feature_filename, test_labels_original_filename, test_translations_filename, scaled_test_features=test_features)
    print "", "\t".join([str(s) for s in scores])
