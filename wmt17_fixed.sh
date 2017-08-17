# leftover experiments with feature scaling for de-en (camera-ready)

# get predictions by baseline for dev
python ../../software/predict.py SVR_organizers2.model features/dev.features dev.mt > predictions/dev.predictions_SVR_organizers2.txt

# get predictions by SLP4 for dev
python ../../software/predict.py MLP_pearson_opti_scale_mlp_edits2.model features/dev.features dev.mt > predictions/dev.predictions_MLP4_scaled_mlp4.txt

# get confidence of SLP4 dev against baseline
python ../../software/confidence.py dev.hter predictions/dev.predictions_SVR_organizers2.txt predictions/dev.predictions_MLP4_scaled_mlp4.txt > dev.SVR_vs_SLP4_clamp.confidence.log

# get confidence of SLP4 test against baseline
python ../../software/confidence.py test.2017.hter predictions/test.2017.predictions_SVR_organizers2.txt predictions/test.2017.predictions_MLP4_scaled_mlp4.txt > test.2017.SVR_vs_SLP4_clamp.confidence.log

# get predictions by SLP1 for dev and test
python ../../software/predict.py MLP_pearson_opti_scale2.model features/dev.features dev.mt > predictions/dev.predictions_MLP_pearson_opti_scale2.txt
python ../../software/predict.py MLP_pearson_opti_scale2.model features/test.2017.features test.2017.mt > predictions/test.2017.predictions_MLP_pearson_opti_scale2.txt
# confidence for SLP1 devset against baseline
python ../../software/confidence.py dev.hter predictions/dev.predictions_MLP_pearson_opti_scale2.txt  predictions/dev.predictions_SVR_organizers2.txt > dev.SLP1_vs_SVR.confidence.log
# confidence for SLP1 testset against baseline
python ../../software/confidence.py test.2017.hter predictions/test.2017.predictions_MLP_pearson_opti_scale2.txt  predictions/test.2017.predictions_SVR_organizers2.txt > test.2017.SLP1_vs_SVR.confidence.log
# confidence for SLP1 devset against SLP4
python ../../software/confidence.py dev.hter predictions/dev.predictions_MLP_pearson_opti_scale2.txt  predictions/dev.predictions_MLP4_scaled_mlp4.txt > dev.SLP1_vs_SLP4.confidence.log
## >> not significant, maybe rescale dev features??
# confidence for testset SLP1 against SLP4
python ../../software/confidence.py test.2017.hter predictions/test.2017.predictions_MLP_pearson_opti_scale2.txt  predictions/test.2017.predictions_MLP4_scaled_mlp4.txt > test.2017.SLP1_vs_SLP4.confidence.log

# run again SVR4
python ../../software/train_test.py 4xSVR_opti_scale features/train.features train.edits features/test.2017.features test.2017.mt test.2017.hter --scale
# get predictions by SVR4 for dev and test
python ../../software/predict.py 4xSVR_opti_scale.model features/dev.features dev.mt --clamp > predictions/dev.predictions_4xSVR_opti_scale.txt
python ../../software/predict.py 4xSVR_opti_scale.model features/test.2017.features test.2017.mt --clamp > predictions/test.2017.predictions_4xSVR_opti_scale.txt
# confidence for devset against MLP4
python ../../software/confidence.py dev.hter predictions/dev.predictions_4xSVR_opti_scale.txt predictions/dev.predictions_MLP4_scaled_mlp4.txt > dev.4xSVR_vs_SLP4_clamp.confidence.log
# confidence for testset against MLP4
python ../../software/confidence.py test.2017.hter predictions/test.2017.predictions_4xSVR_opti_scale.txt predictions/test.2017.predictions_MLP4_scaled_mlp4.txt > test.2017.4xSVR_vs_SLP4_clamp.confidence.log


# Get the optimization function that works best

python ../../software/train_test^Cy MLP_scale_opti_hter_rounded_pearsonr features/train.features train.edits features/test.2017.features test.2017.mt test.2017.hter --scale &> MLP_opti_hter_rounded_pearsonr.log; python ../../software/train_test.py MLP_scale_opti_hter_pearsonr features/train.features train.edits features/test.2017.features test.2017.mt test.2017.hter --scale &> MLP_opti_hter_pearsonr.log; python ../../software/train_test.py MLP_scale_opti_r2 features/train.features train.edits features/test.2017.features test.2017.mt test.2017.hter --scale &> MLP_opti_r2.log

python ../../software/predict.py MLP_scale_opti_r2.model features/dev.features dev.mt --clamp > predictions/dev.predictions_MLP_opti_r2.txt

python ../../software/confidence.py dev.hter predictions/dev.predictions_MLP_opti_r2.txt predictions/dev.predictions_MLP4_scaled_mlp4.txt > MLPr2_vs_MLPpearson_edits.confidence.log

python ../../software/confidence.py dev.hter predictions/dev.predictions_MLP_opti_r2.txt predictions/dev.predictions_MLP4_scaled_mlp4.txt > MLPr2_vs_MLPpearson_edits.confidence.log


# clamp vs round




