python ../../software/train_test.py MLP_pearson_opti advanced_features/train.features train.edits advanced_features/dev.features dev.mt dev.hter
python ../../software/train_test.py MLP_pearson_opti advanced_features/train.features train.hter advanced_features/dev.features dev.mt dev.hter
python ../../software/train_test.py SVR_opti advanced_features/train.features train.hter advanced_features/dev.features dev.mt dev.hter
python ../../software/train_test.py 4xSVR_opti advanced_features/train.features train.edits advanced_features/dev.features dev.mt dev.hter
