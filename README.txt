These are the scripts used for training, development and testing of the experiments for predicting the 4 HTER components jointly with a single layer perceptron. 


Requirements
============

Python 2 requirements can be installed by creating a python virtual environment and then running
pip install -r requirements.txt


Simple overview of scripts
==========================

ter.py:        rerun tercom in order to generate the 4 HTER edits
check_hter:    check the regenerated hter against the hter provided by the organizers
train_test.py: create a model and use it to measure the performance on a development set
predict.py:    apply an existing model to make new predictions
confidence.py: run a significance test between the golden labels and two lists of predictions.

the augmented feature set was produced with Qualitative.
the scripts will be uploaded to a repository, after adding some more documentations and inline comments.
