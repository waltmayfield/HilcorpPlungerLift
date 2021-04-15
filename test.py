import pandas as pd
# import pyodbc
import cx_Oracle

# #Oracle Connection
# cx_Oracle.init_oracle_client(lib_dir=r'C:\Users\wmayfield\instantclient_19_8')
dsn = """(DESCRIPTION =
    (ADDRESS = (PROTOCOL = TCP)(HOST = aztplotdb.hilcorp.com)(PORT = 1521))
    (CONNECT_DATA =
      (SERVER = DEDICATED)
      (SERVICE_NAME = oplosjt1.world)
    )
  )
    """

connection = cx_Oracle.connect("wmayfield", os.environ["ORACLEPSWD"], dsn, encoding="UTF-8")

def loadSQL(sql):#Load Oracle SQL Query
    cursor = connection.cursor()
    cursor.execute(sql)

    column_names = list()
    for i in cursor.description:
        column_names.append(i[0])

    return pd.DataFrame(cursor.fetchall(), columns = column_names)


#Find all wells in the PLOT Database
sql = """
SELECT 
    --Well Identity, DTTM, Cycle_Index
    DISTINCT D.WELLIDA AS UWI
FROM
    PLOT.WELL_LIST_WITH_DATA D
"""

seriesUWIs = loadSQL(sql).UWI

print(seriesUWIs)

# #Server and database info
# server = 'frmsql03.hilcorp.com'
# database = 'SJDW_BI' 

# print("Attempting to initiate connection to server: " + server +", database: " + database)

# #This makes the connection with the server. If there is an error here the connection failed.
# cnxn = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER='+server+';DATABASE='+database+';Trusted_Connection=yes;')
# # cnxn = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER='+server+';DATABASE='+database+';uid=home;pwd=1234;')

# sql = """
# 	SELECT TOP 10 A.AssetCode, A.AssetName
# 	FROM vw_Assets A
# 	"""

# print(sql)

# wells_df = pd.read_sql(sql,cnxn)

# print(wells_df.head())