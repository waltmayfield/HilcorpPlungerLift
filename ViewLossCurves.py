import os, io
import datetime
import pytz
import boto3
import matplotlib.pyplot as plt, pandas as pd
# import fsspec, s3fs

# pd.set_option("display.precision", 0)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:,.0f}'.format

# sHistoryKey = r'LossCurves/2021-01-29-LossCurves.csv'
sHistoryKey = r'LossCurves/2020-12-16-LossCurves.csv'

##################### Authentication #########################################
##This is where the session will look for the profile name
os.environ['AWS_CONFIG_FILE'] = os.path.abspath('./config')
# r"C:\Users\wmayfield\Documents\HilcorpPlungerLift\config"
#r'U:\Projects\ML Plunger Lift Optimizer\.aws\config'

sProfile = 'my-sso-profile-production' #Production version
#print(os.system(f'aws sso login --profile {sProfile}')) ##Comment this out after the first run. Run again if get SSO error.
session = boto3.Session(profile_name=sProfile)

#Experiment sso verification
try:
	print(os.system(f'aws s3 ls --profile {sProfile}'))
else:
	print(os.system(f'aws sso login --profile {sProfile}')) #SSO Log on
################################################################################

bucket_name = 'hilcorp-l48operations-plunger-lift-main'
sHistoryURI = f's3://hilcorp-l48operations-plunger-lift-main/LossCurves/{sHistoryKey}'

s3_client = session.client('s3')

#Download the object using boto3
try:
	obj = s3_client.get_object(Bucket=bucket_name, Key=sHistoryKey)
except:
	print(os.system(f'aws sso login --profile {sProfile}')) #SSO Log on
	obj = s3_client.get_object(Bucket=bucket_name, Key=sHistoryKey)

histDf =pd.read_csv(io.BytesIO(obj['Body'].read()), encoding='utf8').reset_index().drop(['index'], axis = 1)

#Calculated the time since last update
d = datetime.datetime.now()
timezone = pytz.timezone("America/Chicago")
d_aware = timezone.localize(d)
dt = (d_aware - obj['LastModified']).total_seconds()
print('Last update: {:.2f} minutes ago'.format(dt/60))

#print the last few rows of the loss curve csv file
print(histDf.tail(5))

#Create the figure
fig, (ax_mse, ax_loss, ax_acc) = plt.subplots(1, 3, figsize=(25,5))

ax_mse.plot(histDf.index, histDf['val_loss'], label="Validation loss mse")
ax_mse.plot(histDf.index, histDf['loss'], alpha = 0.5, label="Train loss mse")

ax_loss.plot(histDf.index, histDf['val_MCF_metric'], label="Validation MCF mse")
ax_loss.plot(histDf.index, histDf['MCF_metric'], alpha = 0.5, label="Train MCF mse")

ax_acc.plot(histDf.index, histDf["val_plunger_speed_metric"], label="Validation Plunger Speed mse")
ax_acc.plot(histDf.index, histDf["plunger_speed_metric"], alpha = 0.5, label="Train Plunger Speed mse")

fig.suptitle('Path: {}'.format(sHistoryKey))
ax_mse.set_xlabel('Epoch'); ax_mse.set_ylabel('All MSE'); ax_loss.set_xlabel('Epoch'); ax_loss.set_ylabel('MCFD MSE'); ax_acc.set_xlabel('Epoch'); ax_acc.set_ylabel('Plunger Speed MSE')
ax_mse.set_yscale('log'); ax_loss.set_yscale('log'); ax_acc.set_yscale('log'); ax_mse.legend(); ax_loss.legend(); ax_acc.legend()
ax_mse.set_title('Loss Sum'); ax_loss.set_title('MCFD Loss'); ax_acc.set_title('Plunger Speed Loss')
ax_mse.grid(which = 'both'); ax_loss.grid(which = 'both'); ax_acc.grid(which = 'both')

# ax_mse.set_ylim([None,2.9e4]);ax_loss.set_ylim([None,1200]); ax_acc.set_ylim([None,6.0e4])
# ax_acc.set_xlim([0,100])

plt.show()