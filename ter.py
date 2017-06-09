'''
Created on 18 Mar 2014

@author: Eleftherios Avramidis
'''

import tempfile
import os, sys
import subprocess
#from featuregenerator import FeatureGenerator
#from preprocessor import Tokenizer
from xml.sax.saxutils import escape


SGML_CONTAINER = """<refset trglang="en" setid="12014" srclang="any">
<doc sysid="ref" docid="1" genre="news" origlang="xx">
<seg id="1">{text}</seg>
</doc>
</refset>"""


def wrap_sgml(text):
    tmpfile = tempfile.NamedTemporaryFile(mode='w',delete=False,suffix='.jcml', prefix='tmp_', dir='/tmp')
    content = SGML_CONTAINER.format(text=escape(text))
    tmpfile.write(content)
    tmpfile.close()
    return tmpfile.name

class TerWrapper:
    def __init__(self, ter_path, reverse=True):
        self.path = ter_path
        self.reverse = reverse
        
    def process_item(self, target_text, reference_text):
        hypothesis_filename = wrap_sgml(target_text)
        reference_filename = wrap_sgml(reference_text)
        if self.reverse:
            command = "java -jar {} -r {} -h {}".format(self.path, hypothesis_filename, reference_filename)
        else:
            command = "java -jar {} -h {} -r {}".format(self.path, hypothesis_filename, reference_filename)
        
        #java -jar terSimple.jar -h /home/elav01/taraxu_data/wmt14/1.2/tr.translation.seg -r /home/elav01/taraxu_data/wmt14/1.2/tr.suggestion.seg
        
        #print command
        output = subprocess.check_output(command.split())
        atts = {}
        for line in output.split("\n"):
            if line.startswith("RESULT:"):
                atts = dict([item.split(":") for item in line.split()[1:]])
        atts = dict([("ter_{}".format(item[0]), item[1]) for item in atts.iteritems()])
        #os.unlink(hypothesis_filename)
        #os.unlink(reference_filename)        
        
        return atts
    
    def get_features_tgt(self, simplesentence, parallelsentence):
        try:
            return self.process_item(simplesentence.get_string(), parallelsentence.get_reference().get_string())
        except AttributeError:
            sys.stderr.write("Warning: no reference sentences found")
            return {}
        

