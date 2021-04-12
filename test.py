import pyodbc 

server = 
database = 
username = 
password = 


cnxn = pyodbc.connect('DRIVER={/opt/microsoft/msodbcsql17/lib64/libmsodbcsql-17.3.so.1.1};SERVER='+server+';DATABASE='+database+';uid='+username+';pwd='+ password) 


cnxn = pyodbc.connect("Driver={ODBC Driver 17 for SQL Server};"
                      "Server=http://plot-db-production.cluster-c1x26qtu2wyp.us-west-2.rds.amazonaws.com/;"
                      "Database=<DB Name>;"
                      "uid=<username>;pwd=<password>"
                      )
