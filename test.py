import pandas as pd 
import os

dataLoc = r"\\hilcorp.com\Share\Houston\Geology\San Juan Basin\0Personal Work Folders\MLA\For Walt"

for f in os.listdir(dataLoc):
	absPath = os.path.abspath(os.path.join(dataLoc, f))
	print(absPath)

df = pd.read_excel(absPath, engine = 'openpyxl', nrows = 100)

print(df.head())