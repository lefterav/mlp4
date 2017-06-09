from ter import TerWrapper
import sys

PRECISION = 4

#{'ter_deletions': '5', 'ter_shifts': '0', 'ter_ter': '0.6250000000000000', 'ter_substitutions': '0', 'ter_refwords': '8', 'ter_edits': '5', 'ter_insertions': '0'}


if __name__ == '__main__':
    terpath = "/project/qtleap/subprojects/wmt17/qe/software/qualitative/lib/terSimple.jar"
    
    hypothesis_filename = sys.argv[1]
    reference_filename = sys.argv[2]
    scores_filename = sys.argv[3]
    scores_check_filename = sys.argv[4]
    
    
    hypothesis_file = open(hypothesis_filename)
    reference_file = open(reference_filename)
    scores_file = open(scores_filename, 'w')
    scores_file_check = open(scores_check_filename, 'w')
    
    ter = TerWrapper(terpath)
    
    unmatched = 0
    
    i=0
    
    for hypothesis in hypothesis_file:
        i+=1 
        reference = reference_file.readline().strip()

        scores = ter.process_item(hypothesis, reference)
        #print scores
        our_score = round(float(scores["ter_ter"]), PRECISION)
        ter_insertions = float(scores["ter_deletions"])
        ter_deletions = float(scores["ter_insertions"])
        ter_shifts = float(scores["ter_shifts"])
        ter_substitutions = float(scores["ter_substitutions"])

        original_size = len(reference.strip().split(" "))

        new_ter =(ter_insertions + ter_deletions + ter_shifts + ter_substitutions) / (original_size + ter_insertions - ter_deletions)

        print >>scores_file, "{}\t{}\t{}\t{}".format(ter_insertions, ter_deletions, ter_shifts, ter_substitutions)
        print >>scores_file_check, new_ter
    
    print unmatched
    hypothesis_file.close()
    reference_file.close()
    scores_file.close()
    scores_file_check.close()
