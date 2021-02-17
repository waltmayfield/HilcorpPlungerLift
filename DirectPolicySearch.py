#!/usr/bin/env python
# coding: utf-8

import os; import re; import sys
#import importlib

from datetime import datetime

#import matplotlib.pyplot as plt
import boto3
import pandas as pd; import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU, Concatenate
from tensorflow.keras import backend as K

import tqdm

#This is to import local modules
import FunctionsTF as F
#importlib.reload(F) #In case need to reload module after change

pd.set_option("display.precision", 1)


#########################  Parameters ########################
buffer_size = 8
batch_size = 3
###############################################################

print(f'TF version: {tf.__version__}')
lGpus = tf.config.experimental.list_physical_devices('GPU')
print(f'GPUs: {lGpus}')

homeDirectory = r'/EBSPlungerFiles/'
bucket_name = 'hilcorp-l48operations-plunger-lift-main'

#model_name = r'Checkpoints/2020-12-16_460k_Param_LSTM_Skip_resBlock_311Epoch.h5'
model_name = r'Checkpoints/2021-01-29_469472-TrainableVars_LSTM_Skip_resBlock_Larger_MCFD_Leg.h5'

model_save_location = homeDirectory + r'Models/' + model_name

outputPath = homeDirectory + r'RecommendedSettings/' + datetime.today().strftime('%Y-%m-%d') + '-RecommendedSettings.csv'
S3outputPath = f"s3://{bucket_name}/RecommendedSettings/{datetime.today().strftime('%Y-%m-%d')}-RecommendedSettings.csv"
S3outputKey = f"RecommendedSettings/{datetime.today().strftime('%Y-%m-%d')}-RecommendedSettings.csv"

print(f'Output Settings Path: {outputPath}')

TFRecordDirectory = homeDirectory + f'TFRecordFiles/'
print(f'TF Record Files: {[f for f in os.listdir(TFRecordDirectory)]}')

#my_strategy = tf.distribute.MirroredStrategy()

model = load_model(model_save_location, compile = False, custom_objects = {'LeakyReLU' : LeakyReLU()})
print('Model Summary')
print(model.summary())# tf.keras.utils.plot_model(model,show_shapes=True)

s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')


#Get the column names
obj = s3_client.get_object(Bucket = bucket_name, Key= 'DataByAPI/0506705008.csv') 
df = pd.read_csv(obj['Body'], index_col = None, header = 0, dtype = str,nrows = 0)#This loads the column names

#Find index of certain columns
xCols = df.columns[4:].to_list()
ExcessOffTimeIndex = xCols.index('FALL_EST_DIFF_MINS')
ShutInTimeIndex = xCols.index('SHUTIN_LENGTH')
ventIndex = xCols.index('WELL_VENT_SEC')
TimeIndex = xCols.index('FDT')

#lTFRecordFiles = get_ipython().getoutput('ls /home/ec2-user/SageMaker/TFRecordFiles')
lTFRecordFiles = os.listdir(homeDirectory+r'TFRecordFiles/')
lTFRecordFiles =  [homeDirectory + r'TFRecordFiles/' + fName for fName in lTFRecordFiles]

def count_data_items(filenames):
    'Counts the records in each file name'
    n = [int(re.compile(r"-([0-9]*)-Records\.").search(filename).group(1)) for filename in filenames]
#     print(n)
    return np.sum(n)
num_examples = count_data_items(lTFRecordFiles)

raw_dataset = tf.data.TFRecordDataset(lTFRecordFiles)
allWellDs = raw_dataset.map(F.parse_raw_examples_UWI)

# allWellDs = allWellDs.map(lambda x, y, UWI: (x[:100,:],y[:100,:],UWI))#This is just for testing purposes to trim X for shorter computation

allWellDs = allWellDs.prefetch(buffer_size)
allWellDs = allWellDs.map(lambda x, y, UWI: (tf.reverse(x, axis = [0]),tf.reverse(y, axis = [0]),UWI))
allWellDs = allWellDs.padded_batch(batch_size, padded_shapes=([None,79],[None,2],[]))
allWellDs = allWellDs.map(lambda x, y, UWI: (tf.reverse(x, axis = [1]),tf.reverse(y, axis = [1]),UWI))

#for x in tqdm.tqdm_notebook(allWellDs.take(5)): pass

def loss_function(prediction, y):
    rateLoss = -1*prediction[:,-1,0]

    acceptablePlungerSpeeds = y[:,-22:-2,1].mean(axis = 1)*0.99 #Avg of last 20 arrival speeds * factor. The factor allows 1% plunger speed decrease from current value.
    acceptablePlungerSpeeds = np.clip(acceptablePlungerSpeeds,650,800)#No acceptable plunger speed below 650 or above 800 is allowed. These values are cliped to that range

    plungerSpeed = prediction[:,-1,1]
    bUnacceptablePlungerSpeed = (K.cast(K.less(plungerSpeed,acceptablePlungerSpeeds),'float32'))
    plungerLoss = tf.square(plungerSpeed-acceptablePlungerSpeeds)*bUnacceptablePlungerSpeed

    return plungerLoss+rateLoss


#gpu_info = get_ipython().getoutput('nvidia-smi')
#gpu_info = '\n'.join(gpu_info)
#print(gpu_info)

dfSuggestions = pd.DataFrame()
for j, (tX, ty, UWI) in tqdm.tqdm(enumerate(allWellDs), total = int(np.ceil(num_examples/batch_size))):
    # break
    # if j >10: break
    X = tX.numpy()
    y = ty.numpy()
    UWIs = list(map(lambda x: x.decode('utf-8'),UWI.numpy()))

    if X.shape[0] == 0 or X.shape[1] == 0 or len(UWIs)!=X.shape[0]: continue

    # print(UWIs)
    print('X shape: {}'.format(X.shape))
    #This is used to monitor gradient ascent and log changes in loss
    wellIndex = 0 #This well is plotted during gradient ascent
    print('################### Focus API: {}, AVG 20 Plunger Arrival Speed: {:.1f} ####################'.format(UWIs[wellIndex],y[wellIndex,-22:-2,1].mean()))

    # X = X.astype('float32')#This line may not be necessary
    XConstantTensor = tf.constant(X[:,:-1,:])#All of X except the last cycle
    TrainableX = tf.Variable(X[:,-1:,:], trainable = True) #This is just the last cycle of X

    #This will be used to only change the controllable columns
    bMask = np.zeros([X.shape[0],1,X.shape[2]], dtype = np.float32)#Shape is (# wells, 1, # features)
    bMask[:,:,-2:] = 1
    bMaskTensor = tf.constant(bMask)

    #Finish making the X tensor
    XTensor = Concatenate(axis = 1)([XConstantTensor,TrainableX])
    #Save the origional predictions for the next cycle
    yhatOriginal = model(XTensor)[:,-1:,:]

    lr = 1e2
    lossHistory = []
    for i in range(5):#This many gradient loops, increase the # to increase the difference between current and suggested settings
        with tf.GradientTape() as tape:
            XTensor = Concatenate(axis = 1)([XConstantTensor,TrainableX])
            yhat = model(XTensor)
            current_loss = loss_function(yhat,y)  

        #Here is the dLoss/dX calculation
        dX = tape.gradient(current_loss, [TrainableX])[0]

        #Combining the dL/dX and regularization terms
        dX_reg = dX#+regGrad

        # Here I'm clipping the gradient to max of 1 unit change per iteration
        maxUnitsChange = 2.
        dX_reg = tf.clip_by_value(dX_reg,-1*maxUnitsChange/lr,maxUnitsChange/lr)

        #This prevents the actor form changing settigns that controllers don't have.
        dX_masked = tf.math.multiply(dX_reg,bMaskTensor)

        #Here the gradients are applied
        TrainableX = tf.math.subtract(TrainableX,dX_masked*lr)

        #Not sure if need this line
        TrainableX = tf.Variable(TrainableX, trainable = True)

        #This is to plot the result of the settings updates
        productionRate = yhat[wellIndex,-1,0]
        plungerSpeed = yhat[wellIndex,-1,1]

        lossHistory.append(tf.reduce_sum(current_loss))

        if i%2 == 0: 
            print('{:,.2f} Loss Sum,  \t {:,.2f} DMCFD,\t  API:{} Outcome Loss: {:,.2f},    \t DMCFD: {:,.2f},\t Plunger Speed: {:,.2f},\t Csg-Line:{:,.2f},\t CRPct:{:,.2f}'.format(
                tf.reduce_sum(current_loss),
                tf.reduce_sum(yhat[:,-1,0])-tf.reduce_sum(yhatOriginal[:,-1,0]),
                UWIs[wellIndex],
                current_loss[wellIndex],
                yhat[wellIndex,-1,0]-yhatOriginal[wellIndex,-1,0],
                plungerSpeed,
                XTensor.numpy()[wellIndex,-1,-1],#Suggested Csg-Line
                XTensor.numpy()[wellIndex,-1,-2]#Suggested CR Pct End Flow
                ))

    #Now I save the recomended settings
    df = pd.DataFrame(XTensor.numpy()[:,-1,-2:],columns = ['SuggestedCRPctEndFlow','SuggestedCsgMinusLine'])
    df.insert(1,'LastCRPctEndFlow',X[:,-1,-2])
    df.insert(2,'DiffCRPctEndFlow',XTensor.numpy()[:,-1,-2] - X[:,-1,-2])
    df.insert(4,'LastCsgMinusLine',X[:,-1,-1])
    df.insert(5,'DiffCsgMinusLine',XTensor.numpy()[:,-1,-1] - X[:,-1,-1])

    df.insert(6,'AvgExcessOffTime',X[:,-200:-2,ExcessOffTimeIndex].mean(axis = 1)) #If this is 0 then look at changing the plunger drop time to allow the controller to use the suggested Otrig value.
    df.insert(7,'SITimeSTD',X[:,-200:-2,ShutInTimeIndex].std(axis = 1))#If this changes a timer is not controlling the off time
    df.insert(8,'AvgVentSeconds',X[:,-200:-2,ventIndex].mean(axis = 1))#average vent time over last 200 cycles
    df.insert(9,'dLoss/dT',dX[:,0,TimeIndex])#This shows if the model thinks rate goes up or down over time (if the plunger speed is high enough)
    df.insert(10,'LastCycleTime',X[:,-1,TimeIndex])#This is the days between 1/1/1900 and the last cycle
    df.insert(11,'Policy Loss',current_loss)

    df.insert(0,'UWI',UWIs)
    df.insert(1,'DMCFD',yhat[:,-1,0]-yhatOriginal[:,-1,0])
    df.insert(2,'Avg20DMCFD',yhat[:,-1,0]-y[:,-22:-2,0].mean(axis = 1))
    df.insert(3,'PredPSSuggested',yhat[:,-1,1])
    df.insert(4,'PredPSOriginal',yhatOriginal[:,-1,1])
    df.insert(5,'Avg20LastPS',y[:,-22:-2,1].mean(axis = 1))


    dfSuggestions = pd.concat((dfSuggestions,df))
    # break
    # if j > 5: break

print(dfSuggestions.sort_values(by = ['DMCFD'], ascending = False).head(20))


# fig, ((axMCF,axPS),(ax1, ax2)) = plt.subplots(2, 2, figsize=(25,10))

# axMCF.hist(dfSuggestions['DMCFD'],bins=100, alpha=0.5, label = 'Change in Gas Rate')
# axMCF.set_yscale('log')
# axMCF.set_title('Predicted change in gas rate: {:,.2f} MCFD'.format(dfSuggestions['DMCFD'].sum()))
# axMCF.legend()
# axMCF.grid(axis = 'y')
# axMCF.set_xlabel('Change in Gas Rate');axMCF.set_ylabel('Well Count');

# axPS.hist(dfSuggestions['PredPSSuggested'],bins=100, alpha=0.5, label = 'Suggested Policy Predicted Result')
# axPS.hist(dfSuggestions['PredPSOriginal'],bins=100, alpha=0.5, label = 'Current Value')
# axPS.set_yscale('log')
# axPS.set_title('Plunger Speed Histogram')
# axPS.legend()
# axPS.grid(axis = 'y')
# axPS.set(xlim=(0, 1500))#, ylim=(-4.5e3, -2e3))
# axPS.set_xlabel('Plunger Speed');axPS.set_ylabel('Well Count');

# ax1.hist(dfSuggestions['SuggestedCsgMinusLine'].clip(0,200),bins=100, alpha=0.5, label = 'Suggested Value')
# ax1.hist(dfSuggestions['LastCsgMinusLine'].clip(0,200),bins=100, alpha=0.5, label = 'Current Value')
# ax1.set_yscale('log')
# ax1.set_title('CSG-Line Histogram')
# ax1.legend()
# ax1.grid(axis = 'y')
# # ax1.set(xlim=(0, 200))
# ax1.set_xlabel('Casing - Line Pressure Open Trigger');ax1.set_ylabel('Well Count');

# ax2.hist(dfSuggestions['SuggestedCRPctEndFlow'],bins=100, alpha=0.5, label = 'Suggested Value')
# ax2.hist(dfSuggestions['LastCRPctEndFlow'],bins=100, alpha=0.5, label = 'Current Value')
# ax2.set_yscale('log')
# ax2.set_title('Crit Flow Pct End Flow Histogram')
# ax2.legend()
# ax2.grid(axis = 'y')
# # ax.set(xlim=(0, 200))#, ylim=(-4.5e3, -2e3))
# ax2.set_xlabel('Critical Flow Percent Close Trigger');ax2.set_ylabel('Well Count');

dfSuggestions.to_csv(outputPath, index = False)#Save the data frame 

s3_client.upload_file(outputPath,bucket_name,S3outputKey)
