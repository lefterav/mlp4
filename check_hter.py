from ter import TerWrapper
import sys

PRECISION = 4

if __name__ == '__main__':
    terpath = "/project/qtleap/subprojects/wmt17/qe/software/qualitative/lib/terSimple.jar"
    
    hypothesis_filename = sys.argv[1]
    reference_filename = sys.argv[2]
    scores_filename = sys.argv[3]
    
    
    hypothesis_file = open(hypothesis_filename)
    reference_file = open(reference_filename)
    scores_file = open(scores_filename)
    
    ter = TerWrapper(terpath)
    
    unmatched = 0
    
    i=0
    
    for hypothesis in hypothesis_file:
        i+=1 
        reference = reference_file.readline().strip()

        scores = ter.process_item(hypothesis, reference)
        our_score = round(float(scores["ter_ter"]), PRECISION)
        organizers_score = round(float(scores_file.readline().strip()), PRECISION)
        if organizers_score!=our_score:
            print "False:", i, organizers_score, our_score, scores
            unmatched += 1 
            print "\thyp", hypothesis
            print "\tref", reference
            print
        
    
    print unmatched
    hypothesis_file.close()
    reference_file.close()
    scores_file.close()
