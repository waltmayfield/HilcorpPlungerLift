#!/bin/bash
amazon-linux-extras install docker
service docker start
usermod -a -G docker ec2-user
chkconfig docker on
yum install -y git

#This is ment to enable GPU monitoring
pip2.7 install nvidia-ml-py boto3
python2.7 gpumon.py

#Not sure if these will be accessable to ec2-user if made by root
mkdir /home/ubuntu/EBSPlungerFiles
mkdir /home/ubuntu/EBSPlungerFiles/Models
mkdir /home/ubuntu/EBSPlungerFiles/TFRecordFiles
mkdir /home/ubuntu/EBSPlungerFiles/RecommenedSettings
cd /home/ubuntu/EBSPlungerFiles
git clone https://github.com/waltmayfield/HilcorpPlungerLift

cd ../

docker create --gpus all -it --name testContainer --mount type=bind,source=/home/ubuntu/EBSPlungerFiles,target=/EBSPlungerFiles tensorflow/tensorflow:latest-gpu

docker container start testContainer

docker exec testContainer pip install boto3
docker exec testContainer pip install pandas
docker exec testContainer pip install tqdm

docker container stop testContainer

reboot