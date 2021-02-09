
from datetime import datetime; import tempfile, os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, LSTM, Dense, TimeDistributed, Activation, BatchNormalization, Concatenate, LeakyReLU
from tensorflow.keras import regularizers
import boto3
import pydot, graphviz

import FunctionsTF as F

print(f'Tensorflow version: {tf.__version__}')

sModelDescription = r'LSTM_Skip_resBlock_Larger_MCFD_Leg.h5'
dot_img_file = r'U:\Projects\_Output Files/model_1.png'

##################### Authentication #########################################
#This is where the session will look for the profile name
os.environ['AWS_CONFIG_FILE'] = r'U:\Projects\ML Plunger Lift Optimizer\.aws\config'
sProfile = 'my-sso-profile-production' #Production version
#print(os.system(f'aws sso login --profile {sProfile}'))
session = boto3.Session(profile_name=sProfile)
################################################################################

# homeDirectory = f'~/EBSPlungerFiles/'
homeDirectory = tempfile.gettempdir()

bucket_name = 'hilcorp-l48operations-plunger-lift-main'


######################### Model Definition ##############################################
n_channels = 79

regularizer = regularizers.l2(0.01)

inputs = Input(shape = (None,n_channels), dtype = 'float32')

bNorm = BatchNormalization()(inputs)

RUMCF = LSTM(2**5,return_sequences=True, kernel_regularizer = regularizer, recurrent_regularizer = regularizer)(bNorm)

concatMCF = Concatenate(axis = 2)([RUMCF,bNorm])
denseMCF = Dense(2**7,kernel_regularizer = regularizer)(concatMCF)
for _ in range(12):
  # denseMCF = Dense(2**7,kernel_regularizer = regularizer)(denseMCF)
  denseMCF = F.resBlock(denseMCF, l2Reg = 0.01)

outMCF = Dense(1, kernel_regularizer = regularizer)(denseMCF)


RUPS = LSTM(2**3,return_sequences=True, kernel_regularizer = regularizer, recurrent_regularizer = regularizer)(bNorm)

concatPS = Concatenate(axis = 2)([RUPS,bNorm])
densePS = Dense(2**5,kernel_regularizer = regularizer)(concatPS)
for _ in range(3):
  # densePS = Dense(2**7,kernel_regularizer = regularizer)(densePS)
  densePS = F.resBlock(densePS, l2Reg = 0.02)

outPS = Dense(1, kernel_regularizer = regularizer)(densePS)

out = Concatenate(axis = 2)([outMCF,outPS])


####################################### End Model Definition ##################################

model = Model(inputs = inputs, outputs = out)

print('###### Model Summary ########')
print(model.summary())

tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

# from keras.utils import plot_model
# plot_model(model)

# import keras
# import pydot as pyd
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot

# keras.utils.vis_utils.pydot = pyd

# #Visualize Model

# def visualize_model(model):
#   return SVG(model_to_dot(model).create(prog='dot', format='svg'))
# #create your model
# #then call the function on your model
# visualize_model(model)



trainableVars = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
sModelStats = f"{datetime.today().strftime('%Y-%m-%d')}_{trainableVars}-TrainableVars_"
sModelName = sModelStats + sModelDescription
sModelSaveLocation = os.path.join(homeDirectory, sModelName)

print(f'Saving model to {sModelSaveLocation}')

model.save(sModelSaveLocation)

S3outputKey = r"Models/" + sModelName

s3_client = session.client('s3')

print(f'Uploading to bucket: {bucket_name}, key: {S3outputKey}')

s3_client.upload_file(sModelSaveLocation,bucket_name,S3outputKey)