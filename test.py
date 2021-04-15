# import cx_Oracle
import pandas as pd
import pyodbc

#Server and database info
server = 'frmsql03'
database = 'SJDW_BI' 

#Test
sAssetCodes = """
3003907861

"""

FormationPsi =	{
  "FRC": 60,
  "PC": 180,
  "MV": 160,
  "DK": 641,
  "Other": 800
}

sAssetCodes = sAssetCodes.replace(" OR "," ")
lAssetCodes=sAssetCodes.split()
lAssetCodes = ["'{}'".format(x) for x in lAssetCodes]
SelectedAssetCodes = "({})".format(",".join(lAssetCodes))

#This makes the connection with the server. If there is an error here the connection failed.
# cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';Trusted_Connection=yes;')
cnxn = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER='+server+';DATABASE='+database+';uid=home;pwd=1234;')


sql = """
	SELECT A.AssetCode AS AssetCode, MAX(A.AssetName) AS AssetName, MAX(A.ProductionArea) as Area, MAX(A.OperatorRoute) As Route, MAX(A.Formations) AS Formations, MAX(C.InUse) AS WellheadCompInUse, 
	MAX(C.PackageID) as CompPackageID, MAX(C.PackageDescription) as CompPackageDesc, MAX(C.SJEquip_MonthlyRate) AS SJEquip_MonthlyRate, MAX(M.ArtificialLiftType) as ArtificialLiftType, MAX(M.MeterID) as MeterID, M.MeterName as MeterName, Max(M.PipelineCode) as PipelineCode, 
	MAX(M.ProcessingPlant) as ProcessingPlant, COUNT(PDM.TubingPressure) as WellHasTbgP, COUNT(PDM.CasingPressure) as WellHasCsgP
	FROM vw_Assets A
	LEFT JOIN vw_CompressorPackage C ON A.AssetCode = C.AssetCode
	LEFT JOIN vw_Meters M ON M.AssetCode = A.AssetCode
	LEFT JOIN vw_ProductionDailyAllMeters PDM ON PDM.MeterID = M.MeterID AND PDM.ProductionDate = '7/01/2019' AND PDM.CasingPressure > 0
	WHERE A.AssetCode IN {}
	GROUP BY A.AssetCode, M.MeterName
	""".format(SelectedAssetCodes)

print(sql)

wells_df = pd.read_sql(sql,cnxn)

print(wells_df.head())


# cx_Oracle.init_oracle_client(lib_dir=r'C:\Users\wmayfield\instantclient_19_8')
# dsn = """(DESCRIPTION =
#     (ADDRESS = (PROTOCOL = TCP)(HOST = aztplotdb.hilcorp.com)(PORT = 1521))
#     (CONNECT_DATA =
#       (SERVER = DEDICATED)
#       (SERVICE_NAME = oplosjt1.world)
#     )
#   )
#     """

# connection = cx_Oracle.connect("wmayfield", os.environ["ORACLEPSWD"], dsn, encoding="UTF-8")

# def loadSQL(sql):#Load Oracle SQL Query
#     cursor = connection.cursor()
#     cursor.execute(sql)

#     column_names = list()
#     for i in cursor.description:
#         column_names.append(i[0])

#     return pd.DataFrame(cursor.fetchall(), columns = column_names)




# #Find all wells in the PLOT Database
# sql = """
# SELECT 
#     --Well Identity, DTTM, Cycle_Index
#     DISTINCT D.WELLIDA AS UWI
# FROM
#     PLOT.WELL_LIST_WITH_DATA D
# """

# seriesUWIs = loadSQL(sql).UWI