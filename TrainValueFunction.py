
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

import FunctionsTF as F
import Metrics as M

print(f'Tensorflow version: {tf.__version__}')
lGpus = tf.config.experimental.list_physical_devices('GPU')
print(f'GPUs: {lGpus}')
numGPUs = len(lGpus)

######################### Training Parameters ########################
training_epochs = 100
validation_split = 0.1
batch_size = 2*numGPUs
num_parallel_calls = 8
buffer_size = 8
######################################################################

homeDirectory = r'/EBSPlungerFiles/'

model_name = r'2021-01-29_469472-TrainableVars_LSTM_Skip_resBlock_Larger_MCFD_Leg.h5'
model_save_location = homeDirectory + r'Models/' + model_name
#output_model_save_location = homeDirectory + r'Models/' + r'20201216_460k_Param_LSTM_Skip_resBlock.h5'

sBestValLossModelName = model_name[:-3] + '_bestValLoss' + model_name[-3:]
sBestValLossModelLoc = homeDirectory + r'Models/' + sBestValLossModelName

bucket_name = 'hilcorp-l48operations-plunger-lift-main'


S3ModelKey  = f"/Models/{model_name}"
S3BestValLossModelKey = f"/Models/{sBestValLossModelName}"

# historyPath = homeDirectory + r'LossCurves/' + r'20201216History.csv'
historyPathLoc = f"{homeDirectory}/LossCurves/{model_name[:10]}-RecommendedSettings.csv"
historyKeyS3 = f"LossCurves/{model_name[:10]}-RecommendedSettings.csv"

s3_client = boto3.client('s3')

######Remove all files currently in the TF Record Directory
TFRecordDirectory = homeDirectory + f'TFRecordFiles/'
for f in os.listdir(TFRecordDirectory):
    os.remove(os.path.join(TFRecordDirectory, f))

# #Pull up to date Models
# os.system('aws s3 sync s3://hilcorp-l48operations-plunger-lift-main/Models/ ~/EBSPlungerFiles/Models/')
# #Pull up to date Data
# sLatestDataKey=os.popen("aws s3 ls s3://hilcorp-l48operations-plunger-lift-main/TFRecordFiles/ --recursive | sort | tail -n 1 | awk '{print $4}'").read()[:-1]#The -1 removes the new line character
# sS3URILatestDataKey = f"s3://{bucket_name}/{sLatestDataKey}"
# os.system(f'aws s3 cp {sS3URILatestDataKey} ~/EBSPlungerFiles/TFRecordFiles/')

#This creates a new hitory df if one does not already exist
if not os.path.isfile(historyPathLoc):
    print('Creating new history DF at {}'.format(time.localtime()))
    pd.DataFrame(columns = ['loss', 'MCF_metric', 'plunger_speed_metric', 'val_loss', 'val_MCF_metric', 'val_plunger_speed_metric']).to_csv(historyPath, index = False)

#Download the current history path
#dfHistory = pd.read_csv(historyPath)

# strategy = tf.distribute.MirroredStrategy() #This creates a distributed training strategy
# with strategy.scope(): #Models defined withing strategy.scope() are distributed amoung GPUs
model = load_model(model_save_location, compile = False, custom_objects = {'LeakyReLU' : LeakyReLU()})
print('########## Model Summary #############')
print(model.summary())# tf.keras.utils.plot_model(model,show_shapes=True)

## This gets the most recent data file
list_of_files = glob.glob(homeDirectory + f'/TFRecordFiles/*') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime) #This gets the most recently uploaded TF Record File
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
# allWellDs = allWellDs.map(lambda x, y: (x[:100,:],y[:100,:]))#This is just for testing purposes to trim X for shorter computation

trainDs = raw_dataset.skip(numValidWells)
trainDs = trainDs.map(F.parse_raw_examples_UWI, num_parallel_calls=num_parallel_calls)
trainDs = trainDs.map(lambda x, y, UWIs: (x,y))#This is to remove the UWI which is not useful in trianing
trainDs = trainDs.map(lambda x, y: (tf.reverse(x, axis = [0]),tf.reverse(y, axis = [0])))#This is to have leading instead of trailing 0s. Reverse the time direction
trainDs = trainDs.padded_batch(batch_size, padded_shapes=([None,79],[None,2]))# Add the 0s behind the example
trainDs = trainDs.map(lambda x, y: (tf.reverse(x, axis = [1]),tf.reverse(y, axis = [1]))) #Reverse the time to the correct direction
trainDs = trainDs.prefetch(buffer_size)
# trainDs = trainDs.cache(r'./')

validDs = raw_dataset.take(numValidWells)
validDs = validDs.map(F.parse_raw_examples_UWI, num_parallel_calls=num_parallel_calls)
validDs = validDs.map(lambda x, y, UWIs: (x,y))#This is to remove the UWI which is not useful in trianing
validDs = validDs.map(lambda x, y: (tf.reverse(x, axis = [0]),tf.reverse(y, axis = [0])))
validDs = validDs.padded_batch(batch_size, padded_shapes=([None,79],[None,2]))
validDs = validDs.map(lambda x, y: (tf.reverse(x, axis = [1]),tf.reverse(y, axis = [1])))
validDs = validDs.prefetch(buffer_size)


print('Clocking training DS Speed')
for x in tqdm.tqdm(trainDs.take(20)): pass #This is to clock data set speed
print('Clocking validation DS Speed')
for x in tqdm.tqdm(validDs.take(20)): pass #This is to clock data set speed


####### Here are the Check Points ######
class EpochLogger(tf.keras.callbacks.Callback):
    def __init__(self,historyPath):
        self.historyPath = historyPath
        self.historyDf = pd.read_csv(historyPath)
    def on_epoch_end(self,epoch,logs=None):#This saves the loss data
      #I have to save the model here to use the s3 client to upload
      model.save(model_save_location)
      s3_client.upload_file(model_save_location,bucket_name,S3ModelKey)

      current_val_loss = logs.get("val_loss")
      if self.historyDf.shape[0] > 0 and current_val_loss < pd.read_csv(self.historyPath).val_loss.min():

        print(f'New best val_loss score {current_val_loss}. Saving Model to {sBestValLossModelLoc}')
        model.save(sBestValLossModelLoc)
        s3_client.upload_file(sBestValLossModelLoc,bucket_name,S3BestValLossModelKey)
        lossDf = pd.DataFrame(logs, index = [0]) #Turns logs into dataframe
        self.historyDf.append(lossDf)#Append the new row
        s3_client.upload_file(historyPathLoc,bucket_name,historyKeyS3)
        # self.historyDf.to_csv(self.historyPath) #Replace the current loss curve file

# model_checkpoint = ModelCheckpoint(model_save_location, 
#                                    monitor = 'loss', 
#                                    save_best_only=False, 
#                                    save_weights_only = False,
#                                    verbose=1)

terminateOnNaN = keras.callbacks.TerminateOnNaN()
log_results = EpochLogger(historyPath)

optimizer = Adam(lr = 1e-3)
model.compile(loss=M.custom_loss, optimizer=optimizer, metrics = [M.MCF_metric, M.plunger_speed_metric])

steps_per_epoch = int(np.ceil(numTrainWells/batch_size))

model.fit(x = trainDs.repeat(epochs),
          validation_data=validDs,
          epochs = training_epochs,
          steps_per_epoch = steps_per_epoch,
          use_multiprocessing=False,
          callbacks = [
            log_results,
            # model_checkpoint,
            terminateOnNaN
            ]
          )
