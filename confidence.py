from scipy.stats import pearsonr
import random
import time
import sys
from train import load_labels

def compute_confidence(original_labels, predicted_labels_1, predicted_labels_2, samples=1000):

    print pearsonr(predicted_labels_1, original_labels)
    print pearsonr(predicted_labels_2, original_labels)

    label_triples = zip(predicted_labels_1, predicted_labels_2, original_labels)

    # Setting random seed here, to generate same samples for all metrics and directions
    random.seed(int(time.time()))

    rho_1_better = 0
    rho_2_better = 0

    for _ in range(samples):
        label_samples = [random.choice(label_triples) for _ in label_triples]
        predicted_1_sample = [l for l, _, _ in label_samples]
        predicted_2_sample = [l for _, l, _ in label_samples]
        original_sample = [l for _, _, l in label_samples]

        
        rho_1 = pearsonr(predicted_1_sample, original_sample)[0]
        rho_2 = pearsonr(predicted_2_sample, original_sample)[0]
        
        #print rho_1, rho_2
        
        if rho_1 > rho_2:
            rho_1_better+=1
        elif rho_2 > rho_1:
            rho_2_better+=1
    #print rho_1_better
    print "Model 1 is {} better than Model 2".format(1.00*rho_1_better/samples)
    print "Model 2 is {} better than Model 1".format(1.00*rho_2_better/samples)


    
if __name__ == '__main__':
    original_labels = load_labels(sys.argv[1])
    predicted_labels_1 = load_labels(sys.argv[2])
    predicted_labels_2 = load_labels(sys.argv[3])
    compute_confidence(original_labels, predicted_labels_1, predicted_labels_2)
    

