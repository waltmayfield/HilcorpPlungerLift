#!/usr/bin/env python
# coding: utf-8

import os
import boto3
import pandas as pd; import numpy as np
from sklearn.impute import KNNImputer
import tensorflow as tf
from datetime import datetime
import tqdm

print(f'Tensorflow version: {tf.__version__}')

homeDirectory = f'EBSPlungerFiles/'
bucket_name = 'hilcorp-l48operations-plunger-lift-main'
prefix = 'DataByAPI/'

# #sync Data
# os.system('aws s3 sync s3://hilcorp-l48operations-plunger-lift-main/DataByAPI/ ~/EBSPlungerFiles/DataByAPI/')

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

    if outTensor.shape[0]: #Data imputation doesn't work if the input has no rows
        #Now count the non nan values by column. If a column has no non nan values then use a default value
        countNan = tf.cast(tf.math.logical_not(tf.math.is_nan(outTensor)), tf.uint32)
        countNan = tf.math.reduce_sum(countNan, axis = 0)
        emptyColumns = tf.math.equal(countNan,0)
        emptyColBMask = tf.repeat(tf.expand_dims(emptyColumns, axis = 0), repeats = [outTensor.shape[0]], axis = 0)

        # print(f'emptyColBMask: {emptyColBMask.shape}')

        # print(f'countNan shape: {countNan.shape} {countNan}')

        # print(f'empthColumns shape: {emptyColumns.shape} {emptyColumns}')

        #Apply the empty column bmask to outTensor and replace with default value
        outTensor = tf.where(emptyColBMask,100.,outTensor)
        #Use KNN to impute missing data
        print('Imputting Missing Data')
        imputer = KNNImputer(n_neighbors=2)
        outTensor = tf.constant(imputer.fit_transform(outTensor))

    return outTensor

def process_path(file_path):

    # tf.print('file path: {}'.format(file_path))

    PLUNGER_SPEED_loc = 1

    GAS_PER_CYCLE_loc = 23
    FLOW_LENGTH_loc = 12
    SHUTIN_LENGTH_loc = 11
    LEAKING_VALVE_loc = 29

    FLOW_RATE_END_FLOW_loc = 2
    CS_MINUS_LN_SI_loc = 75
    PERCENT_CL_END_FLOW_loc = 76
    inputTensor = csv_to_tensor(file_path)

    print(f'Input Tensor shape: {inputTensor.shape}')

    inputTensor = tf.clip_by_value(inputTensor, -1e6, 1e6, name='ClippedInput')

    #Remove non-physical rows
    bMask = tf.greater(inputTensor[:,SHUTIN_LENGTH_loc],1)
    bMask = tf.logical_and(bMask,tf.greater(inputTensor[:,FLOW_LENGTH_loc],1))
    bMask = tf.logical_and(bMask,tf.greater(inputTensor[:,PLUNGER_SPEED_loc],1))
    bMask = tf.logical_and(bMask,tf.less(inputTensor[:,LEAKING_VALVE_loc],0.5))
    bMask = tf.logical_and(bMask,tf.greater(inputTensor[:,CS_MINUS_LN_SI_loc],1))

    inputTensor = tf.boolean_mask(inputTensor,bMask)

    Xpolicy = tf.stack((
                 inputTensor[1:-1,PERCENT_CL_END_FLOW_loc],
                 inputTensor[2:,CS_MINUS_LN_SI_loc]
                 ),
                 axis = 1)

    X = tf.concat((inputTensor[0:-2,:], #Results
                 Xpolicy), axis = 1) #Next Cycle's controller value

    #This calcualtes the MCFD for the cycle definition which ends with a plunger arrival.
    correctedMCFD = inputTensor[1:-1,GAS_PER_CYCLE_loc]/((inputTensor[1:-1,FLOW_LENGTH_loc]+inputTensor[2:,SHUTIN_LENGTH_loc])/86400.)

    Y = tf.stack((correctedMCFD,
                 inputTensor[2:,1]), #Plunger speed
                 axis = 1,
                #  name = r'Y_'+str(file_path)
                 )

    # X = replaceNanOrInf(X)
    # Y = replaceNanOrInf(Y)

    X = tf.clip_by_value(X,-1e2,1e6)
    Y = tf.clip_by_value(Y,0.,2000.)

    # tf.debugging.check_numerics(X, f'X error, X shape: {X.shape}, file: {file_path} ')
    # tf.debugging.check_numerics(Y, f'Y error, file: {file_path} ')

    return X, Y, file_path


#https://medium.com/radix-ai-blog/tensorflow-sagemaker-d17774417f08
#https://stackoverflow.com/questions/62513518/how-to-save-a-tensor-to-tfrecord
""" 
Converts data to TFRecords file format 
"""

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def convert_ds_to_TFRecord(ds, name, directory):
#     num_elements = ds.reduce(np.int64(0), lambda x, _: x + 1).numpy()
    # filename = os.path.join(directory, f'{name}-{num_elements}-Records.tfrecords')
    tempFileName = os.path.join(directory, r'temp.tfrecords')
    #print(f'Writing {filename}')
    with tf.io.TFRecordWriter(tempFileName) as writer: 
        num_elements = 0
        for X, Y, path in tqdm.tqdm(ds):
            if X.shape[0] < 1000: continue# Don't write stequences with too few time steps
            if min(np.isfinite(X).min(),np.isfinite(Y).min()) == 0:
                print(f'Found NaN or Inf with file : {path}')
                continue

            print(f'Writing example # {num_elements}, shape: {X.shape}')
            num_elements += 1# Add another element to the count
            
            UWI = path.numpy()[-14:-4]
            num_time_steps = X.shape[0]
            # Serialize the tensors
            X_raw = X.numpy().tostring()
            Y_raw = Y.numpy().tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                    'UWI': _bytes_feature(UWI),
                    'X_raw': _bytes_feature(X_raw),
                    'Y_raw': _bytes_feature(Y_raw),
                    'num_time_steps': _int64_feature(num_time_steps)
                    }))

            writer.write(example.SerializeToString())
            
        filename = os.path.join(directory, f'{name}-{num_elements}-Records.tfrecords')
        os.rename(tempFileName,filename)
    return filename


# lFileNames = os.listdir(homeDirectory+r'DataByAPI/')
# num_examples = len(lFileNames)
# print(f'{num_examples} wells to write to file')


if __name__ == "__main__":
    DataFileNames = [homeDirectory + r'DataByAPI/*.csv']
    raw_dataset = tf.data.Dataset.list_files(DataFileNames)

    ######## The take(5) is only for test purposes #######
    allWellDs = raw_dataset.map(process_path)

    ######Remove all files currently in the TF Record Directory
    TFRecordDirectory = homeDirectory + f'TFRecordFiles/'
    for f in os.listdir(TFRecordDirectory):
        os.remove(os.path.join(TFRecordDirectory, f))

    #Add the new .tfrecord file to that directory
    outputFileName = convert_ds_to_TFRecord(allWellDs,f"{datetime.today().strftime('%Y-%m-%d')}_DatasetOneExamplePerWellWithUWI",TFRecordDirectory)

    # !aws s3 cp /home/ec2-user/SageMaker/TFRecordFiles/DatasetOneExamplePerWellWithUWI-5138-Records.tfrecords s3://hilcorp-l48operations-plunger-lift-main/TFRecordFiles/ 
    S3outputKey = outputFileName[len(homeDirectory):]
    print(f'S3outputKey: {S3outputKey}')
    s3_client = boto3.client('s3')
    s3_client.upload_file(outputFileName,bucket_name,S3outputKey)
