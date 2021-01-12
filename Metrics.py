import tensorflow as tf
from tensorflow.keras import backend as K

if __name__ == '__main__':
    print("<> Running the TrainingMetrics as a script! <>")


def custom_loss(y_true,y_pred):
  "This funciton finds the mse between the inputs igonring nan or 0 rows"
  #It's the zero values thats making this not work. They are in both the train but not the validation data sets. I'll remove their loss funciton too
  #This loss function weights the y values with true values higher
  bMask = ~tf.math.is_nan(y_true)
  
  #The y_true can't be zero and the value can't be nan. This is to root out the zero values form the train data set.
  bMask = tf.math.logical_and(bMask,tf.math.not_equal(y_true,0))
 
#   print('bMask: {}'.format(bMask))
  
  #This removes the nan values from y_true and the paired y_pred
  y_true = tf.boolean_mask(y_true,bMask)
  y_pred = tf.boolean_mask(y_pred,bMask)
  
  se = K.square(y_pred - y_true)#Squared Error
  
  output = K.mean(se)
  
  if output != output: 
    # print('y_true type : {}'.format(type(y_true)))
    # print('bMask Shape: {}'.format(bMask.shape))
    # print('sum of bMask:')
    # print(tf.reduce_sum(tf.cast(bMask,tf.float32)).shape)
    
    # if tf.reduce_sum(tf.cast(bMask,tf.float32)) == 0 : print('No values selected in bMask')
    
    # print('NaN Value Encountered in Custom Loss Function')
    output = 0.0
  
  return output
  
  
def MCF_metric(y_true, y_pred):
  'This metric shows preformance on predicting MCFD'
  bMask = ~tf.math.is_nan(y_true[:,:,0])
  
  #The y_true can't be zero and the value can't be nan. This is to root out the zero values form the train data set.
  bMask = tf.math.logical_and(bMask,tf.math.not_equal(y_true[:,:,0],0))
  if tf.reduce_sum(tf.cast(bMask,tf.float32)) == 0: return 0.#If the boolean mask doesn't return any true variales, return 0
  
  #This removes the nan values from y_true and the paired y_pred
  y_true = tf.boolean_mask(y_true[:,:,0],bMask)
  y_pred = tf.boolean_mask(y_pred[:,:,0],bMask)

  se = K.square(y_pred - y_true)#Squared Error

  return K.mean(se)


def plunger_speed_metric(y_true, y_pred):
  'This metric preforms well when the plunger speed is acturately predicted'
  #It's the zero values thats making this not work. They are in both the train but not the validation data sets. I'll remove their loss funciton too
  #This loss function weights the y values with true values higher
  bMask = ~tf.math.is_nan(y_true[:,:,1])
  
  #The y_true can't be zero and the value can't be nan. This is to root out the zero values form the train data set.
  bMask = tf.math.logical_and(bMask,tf.math.not_equal(y_true[:,:,1],0))
  if tf.reduce_sum(tf.cast(bMask,tf.float32)) == 0: return 0.#If the boolean mask doesn't return any true variales, return 0
  
  #This removes the nan values from y_true and the paired y_pred
  y_true = tf.boolean_mask(y_true[:,:,1],bMask)
  y_pred = tf.boolean_mask(y_pred[:,:,1],bMask)
  
  se = K.square(y_pred - y_true)#Squared Error

  return K.mean(se)
  