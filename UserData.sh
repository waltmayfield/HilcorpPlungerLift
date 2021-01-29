#!/bin/bash
amazon-linux-extras install docker
service docker start
usermod -a -G docker ec2-user
chkconfig docker on
yum install -y git

### This is ment to enable GPU monitoring
pip install pynvml
### You need to edit gpumon.py to have the correct region before running the file
python /home/ubuntu/tools/GPUCloudWatchMonitor/gpumon.py &

#Not sure if these will be accessable to ec2-user if made by root
mkdir /home/ubuntu/EBSPlungerFiles
mkdir /home/ubuntu/EBSPlungerFiles/Models
mkdir /home/ubuntu/EBSPlungerFiles/TFRecordFiles
mkdir /home/ubuntu/EBSPlungerFiles/RecommenedSettings
cd /home/ubuntu/EBSPlungerFiles
git clone https://github.com/waltmayfield/HilcorpPlungerLift
cd ../

docker create --gpus all -it --name tfContainer --mount type=bind,source=/home/ubuntu/EBSPlungerFiles,target=/EBSPlungerFiles tensorflow/tensorflow:latest-gpu
#docker create --gpus all -it --name tfContainer --mount type=bind,source=C:\Users\wmayfield\Documents\HilcorpPlungerLift,target=/EBSPlungerFiles tensorflow/tensorflow:latest-gpu

docker container start tfContainer

docker exec tfContainer pip install boto3
docker exec tfContainer pip install pandas
docker exec tfContainer pip install tqdm

docker container stop tfContainer

reboot