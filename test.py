import requests
import base64
import json

ENCODING = 'utf-8'
filePath = r"C:\Users\wmayfield\Desktop\Turdus-migratorius-002.jpg"
fileName = r"Turdus-migratorius-003-python.jpg"

with open(filePath, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())

# print(len(encoded_string))

url = 'https://gv1yedhm7b.execute-api.us-west-2.amazonaws.com/prod/UploadToS3'

myobj = json.dumps({
	'name': fileName,
	'taxonomy': {'species': 'Flamingo'},
	'file': encoded_string.decode(ENCODING)
	},
	indent = 2)

x = requests.post(url, data = myobj)

print(x.text)

### data:image/jpeg;base64,