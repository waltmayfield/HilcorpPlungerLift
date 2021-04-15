import pandas as pd
import pyodbc

#Server and database info
server = 'frmsql03'
database = 'SJDW_BI' 

print(f"Attempting to initiate connection to server: " + server +", database:" + database)

#This makes the connection with the server. If there is an error here the connection failed.
# cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';Trusted_Connection=yes;')
cnxn = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER='+server+';DATABASE='+database+';uid=home;pwd=1234;')

sql = """
	SELECT TOP 10 A.AssetCode, A.AssetName
	FROM vw_Assets A
	"""

print(sql)

wells_df = pd.read_sql(sql,cnxn)

print(wells_df.head())