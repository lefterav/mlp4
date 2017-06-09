# cd en-de; mkdir annotation; mkdir log


# python -u /home/elav01/workspace/qualitative/src/app/hybrid/annotate_batch.py --config ~/workspace/qualitative/config/autoranking/annotation_resources.cfg --sourcelang en --targetlang de --selectpath `pwd` --targetpath `pwd`/annotation --cores 60  &> log/annotation.en-de.20170513.log

# cd en-de/annotation/1

# python /home/elav01/workspace/qualitative/src/support/preprocessing/jcml/jcml2tab/jcmlfeatures2tab.py dev-en-de.all.analyzed.f.jcml selected_features.txt > ../../advanced_features/dev.features

# python /home/elav01/workspace/qualitative/src/support/preprocessing/jcml/jcml2tab/jcmlfeatures2tab.py train-en-de.all.analyzed.f.jcml selected_features.txt > ../../advanced_features/train.features

# cd de-en; mkdir annotation; mkdir log

#python ~/workspace/qualitative/src/support/preprocessing/jcml/txt2jcml.py --source train.src --translation train.mt --reference train.pe --score train.hter --target-features-tab features/train.features --langsrc en --langtgt de -o train.de-en.jcml

#python ~/workspace/qualitative/src/support/preprocessing/jcml/txt2jcml.py --source dev.src --translation dev.mt --reference dev.pe --score dev.hter --target-features-tab features/dev.features --langsrc en --langtgt de -o dev.de-en.jcml

# python -u /home/elav01/workspace/qualitative/src/app/hybrid/annotate_batch.py --config ~/workspace/qualitative/config/autoranking/annotation_resources.cfg --sourcelang de --targetlang en --selectpath `pwd` --targetpath `pwd`/annotation --cores 30  &> log/annotation.de-en.20170513.log

#python ../../software/calculate_hter.py train.pe train.mt train.edits train.hter2
python ../../software/train_test.py SVR_opti features/train.features train.hter features/dev.features dev.mt dev.hter
python ../../software/train_test.py SVR_pearson_opti features/train.features train.hter features/dev.features dev.mt dev.hter
python ../../software/train_test.py 4xSVR_opti features/train.features train.edits features/dev.features dev.mt dev.hter
echo "MLP learning HTER "
python ../../software/train_test.py MLP_opti features/train.features train.hter features/dev.features dev.mt dev.hter
python ../../software/train_test.py MLP_opti features/train.features train.hter features/dev.features dev.mt dev.hter
python ../../software/train_test.py MLP_opti features/train.features train.hter features/dev.features dev.mt dev.hter
echo "MLP learning edits"
python ../../software/train_test.py MLP_opti features/train.features train.edits features/dev.features dev.mt dev.hter
python ../../software/train_test.py MLP_opti features/train.features train.edits features/dev.features dev.mt dev.hter
python ../../software/train_test.py MLP_opti features/train.features train.edits features/dev.features dev.mt dev.hter
echo "MLP learning HTER optimized with Pearson"
python ../../software/train_test.py MLP_pearson_opti features/train.features train.hter features/dev.features dev.mt dev.hter
python ../../software/train_test.py MLP_pearson_opti features/train.features train.hter features/dev.features dev.mt dev.hter
python ../../software/train_test.py MLP_pearson_opti features/train.features train.hter features/dev.features dev.mt dev.hter
echo "MLP learning edits optimized with Pearson"
python ../../software/train_test.py MLP_pearson_opti features/train.features train.edits features/dev.features dev.mt dev.hter
python ../../software/train_test.py MLP_pearson_opti features/train.features train.edits features/dev.features dev.mt dev.hter
python ../../software/train_test.py MLP_pearson_opti features/train.features train.edits features/dev.features dev.mt dev.hter
