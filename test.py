
# import PrepTFRecord as PR
# # import pandas as pd; import numpy as np
# # import tensorflow as tf

# #import os
# import boto3
# import pandas as pd; import numpy as np
# from sklearn.impute import KNNImputer
# import tensorflow as tf
# from datetime import datetime
# import tqdm

import numpy as np


X = np.array([[1.,2.,np.nan],[3.,5.,np.nan],[np.nan, 7., np.nan]])

print(X)

countNan = ~np.isnan(X)
countNan = np.sum(countNan, axis = 0)
emptyColumns = (countNan == 0)
X = np.where(emptyColumns,100,X)

# print(np.where(emptyColumns,-1,X))

# docker run --rm -v U:\scikit-learn-layer:/foo lambci/lambda:build-python3.8 \
# 	pip install scikit-learn -t python

# docker run --rm -v $(pwd)/lambda-layer:/foo lambci/lambda:build-python3.8 \
# 	pip3 install --ignore-installed --target=python scikit-learn

# aws s3 cp scikit-learn.zip s3://hilcorp-l48operations-plunger-lift-main/lambdaDeploymentPackages/ --region us-west-2

# aws lambda publish-layer-version \
#     --layer-name scikit-learn-numpy \
#     --description "Scikit-learn for Python 3.8" \
#     --compatible-runtimes python3.7 python3.8 \
#     --region us-west-2 \
#     --content S3Bucket=hilcorp-l48operations-plunger-lift-main,S3Key=lambdaDeploymentPackages/scikit-learn.zip


# docker run -it --rm -v $(pwd):/app onema/amazonlinux4lambda bash

# docker run -it --rm -v $(pwd):/app lambci/lambda:build-python3.8 bash

# cd /app
# mkdir -p scikitlearn/python
# cd scikitlearn/
# pip3 install --ignore-installed --target=python scikit-learn
# rm -rf python/numpy* python/scipy*
# zip -r ../scikitlearn.zip .

# zip -r scikit-learn.zip python

# aws lambda publish-layer-version  \
# 	--layer-name Python36-SciKitLearn  \
# 	--description "Latest version of scikit learn for python 3.6"  \
# 	--license-info "BSD"  \
# 	--region us-west-2 \
# 	--content S3Bucket=hilcorp-l48operations-plunger-lift-main,S3Key=lambdaDeploymentPackages/scikitlearn.zip  \
# 	--compatible-runtimes python3.6


# fname = r"C:\Users\wmayfield\Downloads\3003921558(2).csv")

# print(pd.read_csv(fname).iloc[:,:10].head())

# print(f'csv shape: {pd.read_csv(fname).iloc[:,4:].shape}')

# # X = PR.csv_to_tensor(fname)
# # print(f'X shape: {X.shape}')
# # print(X[:,:5])
# # print(f'All values finite?:{np.isfinite(X).min()==1} ')

# X, Y, path = PR.process_path(fname)
# if min(np.isfinite(X).min(),np.isfinite(Y).min()) == 0:
# 	print(f'Found NaN or Inf with file : {path}')	
# else:
# 	print('No Nan or Inf')

# print(f'X error location: {np.isfinite(X).argmax()}')

# print(X[0,:])
# print(X.shape)
# print(Y.shape)
# print(path)
# print("Done")

# DataFileNames = [r"C:\Users\wmayfield\Downloads\3003921558(2).csv"]#, r"C:\Users\wmayfield\Downloads\3004534464.csv"]
# #fname = r"C:\Users\wmayfield\Downloads\3004534464.csv"#DataFileNames[0]

# print(pd.read_csv(fname).iloc[:,:10].head())
# print(pd.read_csv(fname).shape)

# X, Y, path = PR.process_path(fname)

# print(f'X shape: {X.shape}')
# print(f'Y shape: {Y.shape}')

# raw_dataset = tf.data.Dataset.list_files(DataFileNames)

# ######## The take(5) is only for test purposes #######
# allWellDs = raw_dataset.map(PR.process_path)

# # ######Remove all files currently in the TF Record Directory
# TFRecordDirectory = r'C:\Users\wmayfield\Downloads'
# # for f in os.listdir(TFRecordDirectory):
# #     os.remove(os.path.join(TFRecordDirectory, f))

# #Add the new .tfrecord file to that directory
# outputFileName = PR.convert_ds_to_TFRecord(allWellDs,f"{datetime.today().strftime('%Y-%m-%d')}_DatasetOneExamplePerWellWithUWI",TFRecordDirectory)

# # !aws s3 cp /home/ec2-user/SageMaker/TFRecordFiles/DatasetOneExamplePerWellWithUWI-5138-Records.tfrecords s3://hilcorp-l48operations-plunger-lift-main/TFRecordFiles/ 
# S3outputKey = outputFileName[len(homeDirectory):]
# print(f'S3outputKey: {S3outputKey}')
# s3_client = boto3.client('s3')
# s3_client.upload_file(outputFileName,bucket_name,S3outputKey)

# print((None or 1))

# import tensorflow as tf

# X = tf.expand_dims(tf.constant([1,2,3,4]), axis = 0)

# print(tf.repeat(X, repeats = [10], axis = 0))


# import numpy as np
# from sklearn.impute import KNNImputer
# X = [[1, 2, np.nan], [3, 4, np.nan], [np.nan, 6, np.nan], [8, 8, np.nan]]
# print(X)
# imputer = KNNImputer(n_neighbors=2)
# print(imputer.fit_transform(X))
