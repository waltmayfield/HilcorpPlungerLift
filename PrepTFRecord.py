#!/usr/bin/env python
# coding: utf-8

import os
import boto3
import pandas as pd; import numpy as np
import tensorflow as tf

import tqdm

print(f'Tensorflow version: {tf.__version__}')

#rootDirectory = '/home/ec2-user/SageMaker/'
homeDirectory = f'~/AttachedVol/EBSPlungerFiles/'
bucket_name = 'hilcorp-l48operations-plunger-lift-main'
prefix = 'DataByAPI/'

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
    colSplit = tf.where(tf.equal(colSplit, b''), b'-1', colSplit)
    #Convert from string to float32
    outTensor = tf.strings.to_number(colSplit, out_type = tf.dtypes.float32, name = 'f32TensorCsv')

    return outTensor

#This is a new path which may be more efficient for training
def replaceNanOrInf(X):
    bMask = tf.math.logical_or(tf.math.is_nan(X),tf.math.is_inf(X))
    return tf.where(bMask,-1.,X)

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


    X = replaceNanOrInf(X)
    Y = replaceNanOrInf(Y)

    X = tf.clip_by_value(X,-1e2,1e6)
    Y = tf.clip_by_value(Y,0.,2000.)

    tf.debugging.check_numerics(X, 'X error, file: {} '.format(file_path))
    tf.debugging.check_numerics(Y, 'Y error, file: {} '.format(file_path))

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

def convert_ds_to_TFRecord(ds, num_elements, name, directory):
#     num_elements = ds.reduce(np.int64(0), lambda x, _: x + 1).numpy()
    filename = os.path.join(directory, f'{name}-{num_elements}-Records.tfrecords')
    print(f'Writing {filename}')
    with tf.io.TFRecordWriter(filename) as writer: 
        for X, Y, path in tqdm.tqdm_notebook(ds):
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



lFileNames = os.listdir(homeDirectory+r'DataByAPI/')
num_examples = len(lFileNames)
print(f'{num_examples} wells to write to file')

DataFileNames = [homeDirectory + r'DataByAPI/*.csv']
raw_dataset = tf.data.Dataset.list_files(DataFileNames)
allWellDs = raw_dataset.map(process_path)

#Remove all files currently in the TF Record Directory
TFRecordDirectory = homeDirectory + f'TFRecordFiles/'
for f in os.listdir(TFRecordDirectory):
    os.remove(os.path.join(TFRecordDirectory, f))

#Add the new .tfrecord file to that directory
convert_ds_to_TFRecord(allWellDs,num_examples,'DatasetOneExamplePerWellWithUWI',TFRecordDirectory)


# !aws s3 cp /home/ec2-user/SageMaker/TFRecordFiles/DatasetOneExamplePerWellWithUWI-5138-Records.tfrecords s3://hilcorp-l48operations-plunger-lift-main/TFRecordFiles/ 
