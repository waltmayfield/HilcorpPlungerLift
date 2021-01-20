Content-Type: multipart/mixed; boundary="//"
MIME-Version: 1.0

--//
Content-Type: text/cloud-config; charset="us-ascii"
MIME-Version: 1.0
Content-Transfer-Encoding: 7bit
Content-Disposition: attachment; filename="cloud-config.txt"

#cloud-config
cloud_final_modules:
- [scripts-user, always]

--//
Content-Type: text/x-shellscript; charset="us-ascii"
MIME-Version: 1.0
Content-Transfer-Encoding: 7bit
Content-Disposition: attachment; filename="userdata.txt"

#!/bin/bash
/bin/echo "Hello World" >> /tmp/testfile.txt
# This code is meant to do the following:
# 1) Pull most recent main branch from https://github.com/waltmayfield/HilcorpPlungerLift
# 2) Install necessary libraries to run direct policy search
# 3) Run the direct policy search which saves a recomendations .csv file in S3
# 4) Close itself down

# Pull Git Repo
#cd ./EBSPlungerFiles/HilcorpPlungerLift
#git pull origin main
#cd ../../

# Start tensorflow container
#docker rm $(docker ps -a -q) #Remove all stopped containers

#GPU
#docker run --gpus all -it --name testContainer --mount type=bind,source="$(pwd)"/EBSPlungerFiles,target=/EBSPlungerFiles tensorflow/tensorflow:latest-gpu

#No GPU
#docker create -it --name testContainer --mount type=bind,source="$(pwd)"/EBSPlungerFiles,target=/EBSPlungerFiles tensorflow/tensorflow:latest

#docker container start testContainer

# Install Necessary Libraries
#docker exec testContainer pip install pandas
#docker exec testContainer pip install boto3
#docker exec testContainer pip install tqdm

# Run direct policy search
# docker exec testContainer python EBSPlungerFiles/HilcorpPlungerLift/DirectPolicySearch.py

# Stop instance
# halt

--//


####### For Error Handeling ########
#!/bin/sh

abort()
{
    echo >&2 '
***************
*** ABORTED ***
***************
'
    echo "An error occurred. Exiting..." >&2
    exit 1
}

trap 'abort' 0

set -e

# Add your script below....
# If an error occurs, the abort() function will be called.
#----------------------------------------------------------
# ===> Your script goes here
# Done!
trap : 0

echo >&2 '
************
*** DONE *** 
************
'