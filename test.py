
import re; import sys; import importlib; import tqdm; import os; import time; import glob
import pathlib
import numpy as np; import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import boto3; import s3fs

# import FunctionsTF as F
# import Metrics as M

#Define the data source
TFRecordDirectory = homeDirectory + f'TFRecordFiles/'

# ## This gets the most recent data file
# list_of_files = glob.glob(homeDirectory + f'TFRecordFiles/*') # * means all if need specific format then *.csv
# latest_file = max(list_of_files, key=os.path.getctime) #This gets the most recently uploaded TF Record File
lTFRecordFiles = [latest_file]
print(f'Most Recent TFRecord File: {lTFRecordFiles}')

def count_data_items(filenames): #This exists incase I want to use multiple .TFRecord files for training
    'Counts the records in each file name'
    n = [int(re.compile(r"-([0-9]*)-Records\.").search(filename).group(1)) for filename in filenames]
#     print(n)
    return np.sum(n)

num_examples = count_data_items(lTFRecordFiles)
numTrainWells = int(np.floor(num_examples*(1.-validation_split)))
numValidWells = num_examples - numTrainWells
print(f"Number of training wells: {numTrainWells}, Validation wells: {numValidWells} of total wells {num_examples}")

########################### This Makes the data set #############################
raw_dataset = tf.data.TFRecordDataset(lTFRecordFiles)
allDs = raw_dataset.map(F.parse_raw_examples_UWI, num_parallel_calls=num_parallel_calls)

for X,Y,UWI in allDs:
	print(X)
	break
# allWellDs = allWellDs.map(lambda x, y: (x[:100,:],y[:100,:]))#This is just for testing purposes to trim X for shorter computation

# trainDs = raw_dataset.skip(numValidWells)
# trainDs = trainDs.map(F.parse_raw_examples_UWI, num_parallel_calls=num_parallel_calls)
# trainDs = trainDs.map(lambda x, y, UWIs: (x,y))#This is to remove the UWI which is not useful in trianing
# trainDs = trainDs.map(lambda x, y: (tf.reverse(x, axis = [0]),tf.reverse(y, axis = [0])))#This is to have leading instead of trailing 0s. Reverse the time direction
# trainDs = trainDs.padded_batch(batch_size, padded_shapes=([None,79],[None,2]))# Add the 0s behind the example
# trainDs = trainDs.map(lambda x, y: (tf.reverse(x, axis = [1]),tf.reverse(y, axis = [1]))) #Reverse the time to the correct direction
# trainDs = trainDs.prefetch(buffer_size)
# # trainDs = trainDs.cache(r'./')

# validDs = raw_dataset.take(numValidWells)
# validDs = validDs.map(F.parse_raw_examples_UWI, num_parallel_calls=num_parallel_calls)
# validDs = validDs.map(lambda x, y, UWIs: (x,y))#This is to remove the UWI which is not useful in trianing
# validDs = validDs.map(lambda x, y: (tf.reverse(x, axis = [0]),tf.reverse(y, axis = [0])))
# validDs = validDs.padded_batch(batch_size, padded_shapes=([None,79],[None,2]))
# validDs = validDs.map(lambda x, y: (tf.reverse(x, axis = [1]),tf.reverse(y, axis = [1])))
# validDs = validDs.prefetch(buffer_size)


# print('Clocking training DS Speed')
# for x in tqdm.tqdm(trainDs.take(20)): pass #This is to clock data set speed
# print('Clocking validation DS Speed')
# for x in tqdm.tqdm(validDs.take(20)): pass #This is to clock data set speed


