import cx_Oracle
import pandas as pd
import csv
import sys
import boto3
from botocore.exceptions import NoCredentialsError
from io import StringIO, BytesIO
import tqdm
import os
import time
from datetime import datetime, timedelta, date
 
#This is where the session will look for the profile name
#os.environ['AWS_CONFIG_FILE'] = r'U:\Projects\ML Plunger Lift Optimizer\.aws\config'
# os.environ['AWS_CONFIG_FILE'] = os.path.abspath('./config')#r"C:\Users\wmayfield\Documents\HilcorpPlungerLift\config"

# sProfile = 'my-sso-profile' #Sandbox Version
# main_bucket_name = 'hilcorp-s3-plungerlift' #Sandbox Version
# bucket_name = 'plunger-lift-temp-data'#Sand box version

# sProfile = 'my-sso-profile-production' #Production version
main_bucket_name = 'hilcorp-l48operations-plunger-lift-main' #Production version
bucket_name = 'hilcorp-l48operations-plunger-lift-temp' #Production version

# sTempFileLoc = r'./tempDf.csv'
csv_buffer = StringIO()

#print(os.system(f'aws sso login --profile {sProfile}'))

session = boto3.Session()#profile_name=sProfile)#.client('sts').get_caller_identity()

main_prefix = 'DataByAPI/'  #This is for the main data set
prefix = 'TempData/'  #This is for new data. It will be added to the main data set with lambda

s3_client = session.client('s3')
s3_resource = session.resource('s3')

#APIs already in bucket
my_bucket = s3_resource.Bucket(bucket_name)
allObjects = my_bucket.objects.all()

lApisInBucket = [o.key[len(prefix):-4] for o in allObjects if o.key[0:len(prefix)] == prefix]

#Oracle Connection
# cx_Oracle.init_oracle_client(lib_dir=r'C:\Users\wmayfield\instantclient_19_8')
dsn = """(DESCRIPTION =
    (ADDRESS = (PROTOCOL = TCP)(HOST = aztplotdb.hilcorp.com)(PORT = 1521))
    (CONNECT_DATA =
      (SERVER = DEDICATED)
      (SERVICE_NAME = oplosjt1.world)
    )
  )
    """

# print(f'Environmental variables: {os.environ}')

# connection = cx_Oracle.connect("wmayfield", os.environ["ORACLEPSWD"], dsn, encoding="UTF-8")
connection = cx_Oracle.connect("hecplot", "PL0tmy_da7a", dsn, encoding="UTF-8")

def loadSQL(sql):#Load Oracle SQL Query
    cursor = connection.cursor()
    cursor.execute(sql)

    column_names = list()
    for i in cursor.description:
        column_names.append(i[0])

    return pd.DataFrame(cursor.fetchall(), columns = column_names)


# #APIs already in bucket
# my_bucket = s3_resource.Bucket(bucket_name)
# allObjects = my_bucket.objects.all()
# lApisInBucket = [o.key[len(prefix):-4] for o in allObjects if o.key[0:len(prefix)] == prefix]
# lApisInBucket = [a for a in lApisInBucket if len(a) == 10]#This knocks out non API# files in the S3 bucket

#Find all wells in the PLOT Database
sql = """
SELECT 
    --Well Identity, DTTM, Cycle_Index
    DISTINCT D.WELLIDA AS UWI
FROM
    PLOT.WELL_LIST_WITH_DATA D
"""

seriesUWIs = loadSQL(sql).UWI

# print('# UWIs before eliminate those already in bucket: {}'.format(seriesUWIs.shape))

# #Don't upload if API already in bucket
# seriesUWIs = seriesUWIs[~seriesUWIs.isin(lApisInBucket)]

# print('After: {}'.format(seriesUWIs.shape))

#Don't pull wells from today to avoid incomplete cycles
dateToday = date.today() + timedelta(days=-1)
sEndDate = str(dateToday.year)+ "-" + str(dateToday.month) + "-" + str(dateToday.day)

print(f'Last Date To Pull Data From: {sEndDate}')
# seriesUWIs = ['0506705009']

for i, UWI in tqdm.tqdm(seriesUWIs.items()):
    # time.sleep(100)

    # UWI = '3003926762'#This is for testing

    try:
        obj = s3_client.get_object(Bucket=main_bucket_name, Key=main_prefix + str(UWI) + '.csv')
        lastDate = obj['LastModified'].date() + timedelta(days=-7)#It's OK to pull a small overlap. The overlap will not be written to the main file
        sStartDate = str(lastDate.year) + "-" + str(lastDate.month) + "-" + str(lastDate.day)
        # iSecondsSinceUpdate = (obj['LastModified'].date()-lastDate).total_seconds()
        # # print('Days since update: {}'.format(iSecondsSinceUpdate/(86400)))
        # if iSecondsSinceUpdate < 86400*5: #Don't update if the file was changed less than 5 days ago
        #     continue #If the file was recently updated
    except:
        sStartDate = "2010-01-01"


    sql = """
    SELECT 
        --Well Identity, DTTM, Cycle_Index
        D.WELLIDA AS UWI, D.WELLNAME, C.CYCLE_INDEX, TO_CHAR(C.START_TIME, 'YYYY-MM-DD HH24:MI:SS') AS StartTime,
        
        --These are the values to predict
        C.GAS_PER_DAY, 
        --If the plunger doesn't arrive, then set plunger arival speed as 0.
        CASE C.PLUNGER_ARRIVAL_FLAG WHEN 'Y' THEN C.PLUNGER_SPEED ELSE 0.0 END AS PLUNGER_SPEED, 

        --These columns may be used as features to control in the future   
        C.FLOW_RATE_END_FLOW,
        C.AFTERFLOW_MINS, 
        
        --Here are the observable columns included in the analysis
        C.CL_END_FLOW, C.TUBING_PRESSURE_AVG_FLOW, C.CASING_PRESSURE_AVG_FLOW,
        C.WELL_VENT, C.LOAD_RATIO_PRIOR_OPEN, C.CSG_OVER_LINE, C.CYCLE_LENGTH, C.SHUTIN_LENGTH, C.FLOW_LENGTH, C.CAS_TUB_DIFF_MAX, C.CAS_TUB_DIFF_AVG, 
        C.CAS_TUB_DIFF_MIN, C.CAS_TUB_DIFF_LAST, C.TUB_LINE_DIFF_MAX, C.TUB_LINE_DIFF_AVG, C.TUB_LINE_DIFF_MIN, C.AVG_BHP, C.AVG_BHP_SI, C.AVG_BHP_FLOW, C.GAS_PER_CYCLE, 
        C.PLUNGER_TRAVEL_TIME, C.AVG_SLUG_HEIGHT, C.WELL_VENT_SEC, C.MIN_DP, C.MAX_DP, C.LEAKING_VALVE, C.LIFT_MINS, C.PLUNGER_FALL_TIME,
        C.AVG_TUB_MIN_LINE_FLOW_EXCESS, C.CASING_FLAT_MINS, C.FALL_EST_DIFF_MINS, C.PLUNGER_FALL_EXCESS, C.CSG_TUB_DIFF_MAX_SI, C.AVG_DP_FLOW, C.SLUG_ARRIVAL,
        C.SLUG_SIZE_LT_MAX, C.ARRIVAL_SENSOR_FAIL, C.CONSEC_NON_ARRIVALS, C.CAS_TUB_DIFF_MAX_SI, C.MAX_SLUG_SIZE, C.TUB_LT_LINE_SI_MINS, C.DP_FLOW_GT_115_MINS,
        C.DP_FLOW_GT_MACH_MINS, C.TUB_LINE_DIFF_MAX_SI, C.PLUNGER_FALL_EST_LIQUID, C.PLUNGER_FALL_EST_GAS, C.PLUNGER_FALL_EST_TOTAL_CSN, C.CSG_BUILD_RATE,
        C.SLUG_CHANGE_AT_OPEN, C.CSG_RISE_TOTAL_SI, C.LINE_PRESSURE_AVG_FLOW, C.SLUG_RATE_PER_DAY, C.OFF_TIME_LONG_PCENT, C.SLUG_SIZE_AT_OPEN, C.SLUG_PCENT_MAX_OPEN,
        C.TUB_CAS_ANNULUS_BRIDGE, C.PERCENT_ON_TIME_VENTING, C.LINE_PRESSURE_AVG_SI, C.CASING_DIFF, C.LINE_PRESSURE_JUMP_AT_OPEN, C.ARRIVAL_FAST_DRY, C.AVG_CSG_MIN_LINE,
        C.AVG_CSG_MIN_LINE_SHUTIN, C.GAS_IN_TUBING_BEFORE_OPEN, C.GAS_PRODUCED_BEFORE_ARRIVAL, C.GAS_RATIO_PRODUCED_OVER_STORED, 
        
        ---- This is the days from 1/1/1900 to RowDate in float32 format
        CAST( (C.START_TIME - TO_DATE('1900-01-01', 'YYYY-MM-DD')) AS FLOAT(32)) as fDT,
        
        ---- This is the days from 1/1/1900 to SpudDate in float32 format
        CAST( (TO_DATE(D.DTTMSPUD, 'DD-MON-RR') - TO_DATE('1900-01-01', 'YYYY-MM-DD')) AS FLOAT(32)) as fSpudDate,
        
        ---- Here are various well header columns
        WD.CASING_ID, WD.TUBING_ID, WD.TUBING_LENGTH_FT,
        
        --These are the columns to control
        C.CSG_LINE_DIFF_MAX_SI, C.PERCENT_CL_END_FLOW
        
    FROM PLOT.WBORE_CYCLE_CALC C
    LEFT JOIN PLOT.WELL_LIST_WITH_DATA D ON D.PLOT_WB_ID = C.PLOT_WB_ID
    LEFT JOIN PLOT.WBORE_DOWNHOLE_SECTIONS WD ON WD.PLOT_WB_ID = C.PLOT_WB_ID
    WHERE C.IS_PARTIAL_CYCLE = 0

    -- AND C.DATE_CALC <  DATE '2020-11-14'
    AND C.DATE_CALC <=  DATE '{}'
    AND C.DATE_CALC >= DATE '{}'

    AND C.PERCENT_CL_END_FLOW IS NOT NULL --This will be removed in future versions
    AND D.WELLIDA = '{}'
    --AND ROWNUM < 100
    ORDER BY C.PLOT_WB_ID, C.START_TIME
    """.format(sEndDate,sStartDate,UWI)

    tempDf = loadSQL(sql)

    if tempDf.shape[0] < 1:
        continue#This is in case there is no data for a well

    # tempDf.to_csv(sTempFileLoc, index = False)
    # s3_client.upload_file(sTempFileLoc,bucket_name,prefix + '{}.csv'.format(UWI))

    s3Key = prefix + '{}.csv'.format(UWI)
    # s3Key = prefix + 'testWellFile.csv'

    print(f"Uploading well # {i} to bucket {bucket_name} and key {s3Key} for data between {sStartDate} and {sEndDate}")

    tempDf.to_csv(csv_buffer, index = False)
    s3_resource.Object(bucket_name, s3Key).put(Body = csv_buffer.getvalue())

    csv_buffer.truncate(0) #Likely not necessary but it makes me feel safer. Layered security right?
    # print(UWI)
    # break
# os.remove(sTempFileLoc)

# #Now wait 20 minutes for the lambdas to complete and run the policy search step function
# print("Now waiting 20 mintues for data prep lambdas to complete before initiating step fuction")
# for i in tqdm.tqdm(range(20)):
#     time.sleep(60)

# #Create a step function client
# SFNclient = session.client('stepfunctions', region_name = 'us-west-2')

# #Start the step function to make plunger lift settings recommendations
# response = SFNclient.start_execution(
#     stateMachineArn='arn:aws:states:us-west-2:446356438225:stateMachine:PlungerPolicySearch'
# )

# print(response)