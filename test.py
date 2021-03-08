import pyodbc 

# import os, io
# import datetime
# import pytz
# import boto3
# import matplotlib.pyplot as plt, pandas as pd
# # import fsspec, s3fs

# # pd.set_option("display.precision", 0)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# pd.options.display.float_format = '{:,.0f}'.format

# # sHistoryKey = r'LossCurves/2021-01-29-LossCurves.csv'
# sHistoryKey = r'LossCurves/2020-12-16-LossCurves.csv'

# ##################### Authentication #########################################
# ##This is where the session will look for the profile name
# os.environ['AWS_CONFIG_FILE'] = os.path.abspath('./config')
# # r"C:\Users\wmayfield\Documents\HilcorpPlungerLift\config"
# #r'U:\Projects\ML Plunger Lift Optimizer\.aws\config'

# sProfile = 'my-sso-profile-production' #Production version
# #print(os.system(f'aws sso login --profile {sProfile}')) ##Comment this out after the first run. Run again if get SSO error.
# session = boto3.Session(profile_name=sProfile)
# ################################################################################

# bucket_name = 'hilcorp-l48operations-plunger-lift-main'
# sHistoryURI = f's3://hilcorp-l48operations-plunger-lift-main/LossCurves/{sHistoryKey}'

# s3_client = session.client('s3')

# #Download the object using boto3
# # try:
# # 	obj = s3_client.get_object(Bucket=bucket_name, Key=sHistoryKey)
# # except:
# # 	print(os.system(f'aws sso login --profile {sProfile}')) #SSO Log on
# # 	obj = s3_client.get_object(Bucket=bucket_name, Key=sHistoryKey)

# print(os.system(f'aws s3 ls --profile {sProfile}'))
# print('hello world')

# print(os.system(f'aws sso login --profile {sProfile}'))
# print(os.system('aws s3api list-buckets --query "Buckets[].Name"'))
# print(os.system(f'aws sso login --profile {sProfile}'))

# try:
#     lApisInBucket = [o.key[len(prefix):-4] for o in allObjects if o.key[0:len(prefix)] == prefix]
# except:
#     print(os.system(f'aws sso login --profile {sProfile}'))
#     lApisInBucket = [o.key[len(prefix):-4] for o in allObjects if o.key[0:len(prefix)] == prefix]



# engine = psycopg2.connect(
#     database="postgres",
#     user="my_user_name",
#     password="abc123def345",
#     host="plot-db-production.cluster-ro-c1x26qtu2wyp.us-west-2.rds.amazonaws.com",
#     port='5432'
# )

# print('done')
