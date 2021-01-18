
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, LSTM, Dense, TimeDistributed, Activation, BatchNormalization, Concatenate, LeakyReLU
from tensorflow.keras import regularizers
import FunctionsTF as F


print(f'Tensorflow version: {tf.__version__}')


homeDirectory = f'~/EBSPlungerFiles/'
sModelDescription = r'LSTM_Skip_resBlock.h5'

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


RUPS = LSTM(2**4,return_sequences=True, kernel_regularizer = regularizer, recurrent_regularizer = regularizer)(bNorm)

concatPS = Concatenate(axis = 2)([RUPS,bNorm])
densePS = Dense(2**6,kernel_regularizer = regularizer)(concatPS)
for _ in range(3):
  # densePS = Dense(2**7,kernel_regularizer = regularizer)(densePS)
  densePS = F.resBlock(densePS, l2Reg = 0.01)

outPS = Dense(1, kernel_regularizer = regularizer)(densePS)

out = Concatenate(axis = 2)([outMCF,outPS])

model = Model(inputs = inputs, outputs = out)

print('###### Model Summary ########')
print(model.summary())

trainableVars = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
sModelStats = f"{datetime.today().strftime('%Y-%m-%d')}_{trainableVars}-TrainableVars_"
sModelName = sModelStats + sModelDescription
sModelSaveLocation = homeDirectory + r'Models/' + sModelName

print(f'Saving model to {sModelSaveLocation}')

model.save(sModelSaveLocation)

### C:\Users\wmayfield\Documents\HilcorpPlungerLift

### docker create -v C:\Users\wmayfield\Documents\HilcorpPlungerLift:/data --name testContainer tensorflow/tensorflow:latest

### docker create --name testContainer tensorflow/tensorflow:latest

### docker run -it -v C:\Users\wmayfield\Documents\HilcorpPlungerLift:/data --name testContainer tensorflow/tensorflow:latest bash