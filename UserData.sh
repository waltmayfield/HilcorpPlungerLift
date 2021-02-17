#!/bin/bash
#The deep learning Ubuntu AMIs alread have docker and git
amazon-linux-extras install docker
service docker start
usermod -a -G docker ec2-user
chkconfig docker on
yum install -y git

### This is ment to enable GPU monitoring
pip install pynvml
### You need to edit gpumon.py to have the correct region before running the file. The logs will be under DeepLearningTrain metric group
python /home/ubuntu/tools/GPUCloudWatchMonitor/gpumon.py &

#Not sure if these will be accessable to ec2-user if made by root
mkdir /home/ubuntu/EBSPlungerFiles
mkdir /home/ubuntu/EBSPlungerFiles/Models
mkdir /home/ubuntu/EBSPlungerFiles/TFRecordFiles
mkdir /home/ubuntu/EBSPlungerFiles/RecommendedSettings
mkdir /home/ubuntu/EBSPlungerFiles/LossCurves

# Clone git repository
cd /home/ubuntu/EBSPlungerFiles
git clone https://github.com/waltmayfield/HilcorpPlungerLift
cd ../

#Download most recent data file
aws s3 cp s3://hilcorp-l48operations-plunger-lift-main/"$(aws s3 ls s3://hilcorp-l48operations-plunger-lift-main/TFRecordFiles/ --recursive | sort | tail -n 1 | awk '{print $4}')" /home/ubuntu/EBSPlungerFiles/TFRecordFiles/


docker create --gpus all -it --name tfContainer --mount type=bind,source=/home/ubuntu/EBSPlungerFiles,target=/EBSPlungerFiles tensorflow/tensorflow:latest-gpu
#docker create --gpus all -it --name tfContainer --mount type=bind,source=C:\Users\wmayfield\Documents\HilcorpPlungerLift,target=/EBSPlungerFiles tensorflow/tensorflow:latest-gpu

docker container start tfContainer 

docker exec tfContainer pip install -U scikit-learn
docker exec tfContainer pip install boto3
docker exec tfContainer pip install pandas
docker exec tfContainer pip install tqdm
#These are for pandas to interact with s3
docker exec tfContainer pip install fsspec
docker exec tfContainer pip install awscli
docker exec tfContainer pip install s3fs

docker container stop tfContainer

reboot

aws s3 cp s3://hilcorp-l48operations-plunger-lift-main/"$(aws s3 ls s3://hilcorp-l48operations-plunger-lift-main/TFRecordFiles/ --recursive | sort | tail -n 1 | awk '{print $4}')" /home/ubuntu/EBSPlungerFiles/TFRecordFiles/
docker container start tfContainer 
docker exec tfContainer python EBSPlungerFiles/HilcorpPlungerLift/TrainValueFunction.py
