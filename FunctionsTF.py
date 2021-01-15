from tensorflow import keras
import tensorflow as tf
import numpy as np, os

if __name__ == '__main__':
    print("<> Running the module as a script! <>")

# def parse_raw_examples(record):
#   features = {
#           'X_raw': tf.io.FixedLenFeature([], tf.string, default_value='-1'),
#           'Y_raw': tf.io.FixedLenFeature([], tf.string, default_value='-1'),
#           'num_time_steps': tf.io.FixedLenFeature([], tf.int64)
#       }
      
#   n_features = 79
#   parsed = tf.io.parse_single_example(record, features)

#   n_time_steps = parsed['num_time_steps']

#   X_dec= tf.io.decode_raw(parsed['X_raw'], tf.float32)
#   X = tf.reshape(X_dec,(n_time_steps,n_features))

#   Y_dec= tf.io.decode_raw(parsed['Y_raw'], tf.float32)
#   Y = tf.reshape(Y_dec,(n_time_steps,2))

#   return (X,Y)
 
def parse_raw_examples_UWI(record):
  features = {
          'UWI': tf.io.FixedLenFeature([], tf.string, default_value='-1'),
          'X_raw': tf.io.FixedLenFeature([], tf.string, default_value='-1'),
          'Y_raw': tf.io.FixedLenFeature([], tf.string, default_value='-1'),
          'num_time_steps': tf.io.FixedLenFeature([], tf.int64)
      }
      
  n_features = 79
  parsed = tf.io.parse_single_example(record, features)

  n_time_steps = parsed['num_time_steps']
  UWI = parsed['UWI']

  X_dec= tf.io.decode_raw(parsed['X_raw'], tf.float32)
  X = tf.reshape(X_dec,(n_time_steps,n_features))

  Y_dec= tf.io.decode_raw(parsed['Y_raw'], tf.float32)
  Y = tf.reshape(Y_dec,(n_time_steps,2))

  return (X,Y, UWI)

#This is a new path which may be more efficient for training
def csv_to_tensor(file_path):
  #First read the file into a string
  sCsv = tf.io.read_file(file_path)#, [tf.constant('-1')]*1)#.numpy().decode('utf-8')
  #Then split the string into rows
  rowSplit = tf.strings.split(sCsv, sep=b'\r\n', maxsplit=-1, name='SplitLines')
  rowSplit = tf.strings.split(sCsv, sep=b'\n', maxsplit=-1, name='SplitLines')#Some lines use the /n w/0 the /r
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

# def process_path(file_path):

#   # tf.print('file path: {}'.format(file_path))
  
#   PLUNGER_SPEED_loc = 1
  
#   GAS_PER_CYCLE_loc = 23
#   FLOW_LENGTH_loc = 12
#   SHUTIN_LENGTH_loc = 11
#   LEAKING_VALVE_loc = 29
    
#   FLOW_RATE_END_FLOW_loc = 2
#   CS_MINUS_LN_SI_loc = 75
#   PERCENT_CL_END_FLOW_loc = 76
#   inputTensor = csv_to_tensor(file_path)

#   inputTensor = tf.clip_by_value(inputTensor, -1e6, 1e6, name='ClippedInput')
  
#   #Remove non-physical rows
#   bMask = tf.greater(inputTensor[:,SHUTIN_LENGTH_loc],1)
#   bMask = tf.logical_and(bMask,tf.greater(inputTensor[:,FLOW_LENGTH_loc],1))
#   bMask = tf.logical_and(bMask,tf.greater(inputTensor[:,PLUNGER_SPEED_loc],1))
#   bMask = tf.logical_and(bMask,tf.less(inputTensor[:,LEAKING_VALVE_loc],0.5))
#   bMask = tf.logical_and(bMask,tf.greater(inputTensor[:,CS_MINUS_LN_SI_loc],1))

#   inputTensor = tf.boolean_mask(inputTensor,bMask)
  
#   Xpolicy = tf.stack((
#                  inputTensor[2:,CS_MINUS_LN_SI_loc],
#                  inputTensor[1:-1,PERCENT_CL_END_FLOW_loc]
#                  ),
#                  axis = 1)

#   X = tf.concat((inputTensor[0:-2,:], #Results
#                  Xpolicy), axis = 1) #Next Cycle's controller value

#   #This calcualtes the MCFD for the cycle definition which ends with a plunger arrival.
#   correctedMCFD = inputTensor[1:-1,GAS_PER_CYCLE_loc]/((inputTensor[1:-1,FLOW_LENGTH_loc]+inputTensor[2:,SHUTIN_LENGTH_loc])/86400.)
 
#   Y = tf.stack((correctedMCFD,
#                  inputTensor[2:,1]), #Plunger speed
#                  axis = 1,
#                 #  name = r'Y_'+str(file_path)
#                  )


#   X = replaceNanOrInf(X)
#   Y = replaceNanOrInf(Y)

#   X = tf.clip_by_value(X,-1e2,1e6)
#   Y = tf.clip_by_value(Y,0.,2000.)

#   tf.debugging.check_numerics(X, 'X error, file: {} '.format(file_path))
#   tf.debugging.check_numerics(Y, 'Y error, file: {} '.format(file_path))

#   return X, Y#, file_path

def process_path_UWI(file_path):

  tf.print('file path: {}'.format(file_path))
  
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
                 inputTensor[2:,CS_MINUS_LN_SI_loc],
                 inputTensor[1:-1,PERCENT_CL_END_FLOW_loc]
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
  
def resBlock(input_layer, l2Reg = 0.01):
  'This residual block takes any 1D input and preformes the block calculations'
  from tensorflow.keras.layers import Dense, Activation, BatchNormalization, LeakyReLU
  from tensorflow.keras import regularizers
  import tensorflow as tf
  denseUnits = input_layer.shape[2]#                          Length of 1D input
  skipCxn = input_layer#                                      saved to be added later
  dense = Dense(denseUnits, activation = None, kernel_regularizer=regularizers.l2(l2Reg))(input_layer)#  First Dense Layer
  bNorm = BatchNormalization()(dense)#                        Batch Normilization
  activated = Activation(LeakyReLU())(bNorm)#                   Activation Layer
  dense = Dense(denseUnits, activation = None, kernel_regularizer=regularizers.l2(l2Reg))(activated)#    Second dense layer
  bNorm = BatchNormalization()(dense)#                        Batch Normilztion
  activated = Activation(LeakyReLU())(bNorm)#                   Activate the state
  added = tf.add(activated, skipCxn)#                         Add input to current state
  output_layer = added
  return output_layer