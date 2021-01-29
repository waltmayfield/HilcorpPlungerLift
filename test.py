
import os
import boto3
import pandas as pd; import numpy as np
from sklearn.impute import KNNImputer
import tensorflow as tf
from datetime import datetime
import tqdm

pd.set_option("display.precision", 1)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#This is a new path which may be more efficient for training
def replaceNanOrInf(X):
    bMask = tf.math.logical_or(tf.math.is_nan(X),tf.math.is_inf(X))
    return tf.where(bMask,-1.,X)

#This is a new path which may be more efficient for training
def csv_to_tensor(file_path):
    #First read the file into a string
    sCsv = tf.io.read_file(file_path)#, [tf.constant('-1')]*1)#.numpy().decode('utf-8')
    #Then split the string into rows
    rowSplit = tf.strings.split(sCsv, sep=b'\r\n', maxsplit=-1, name='SplitLines')
    rowSplit = tf.strings.split(sCsv, sep=b'\n', maxsplit=-1, name='SplitLines')
    #The split that into rows and columns
    colSplit = tf.strings.split(rowSplit, sep=b',', maxsplit=-1, name='SplitLines')
    #Remove the header. The last line will be empty b/c the previous ended with the new line character. Convert to tensor
    colSplit = colSplit[1:-1,4:].to_tensor()#This removes the column header and the empty last row

    #Replace empty strings with -1
    colSplit = tf.where(tf.equal(colSplit, b''), b'NaN', colSplit)
    
    #Convert from string to float32
    outTensor = tf.strings.to_number(colSplit, out_type = tf.dtypes.float32, name = 'f32TensorCsv')

    # print(f'################### outTensor {outTensor.shape} {outTensor}')

    if outTensor.shape[0]:
        #Use KNN to impute missing data
        imputer = KNNImputer(n_neighbors=2)
        outTensor = tf.constant(imputer.fit_transform(outTensor))
    
    # #Replace missing values with imputed values
    # outTensor = tfp.sts.MaskedTimeSeries(
    #     time_series=outTensor,
    #     is_missing=replaceNanOrInf(outTensor))

    return outTensor

fname = r"C:\Users\wmayfield\Downloads\3003921558(2).csv"
print(pd.read_csv(fname).iloc[:,:10].head())

print(csv_to_tensor(fname)[:5,:10])

print("Done")