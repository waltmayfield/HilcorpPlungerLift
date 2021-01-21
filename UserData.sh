#!/bin/bash
sudo amazon-linux-extras install docker
sudo service docker start
sudo usermod -a -G docker ec2-user
sudo chkconfig docker on
sudo yum install -y git
sudo reboot

mkdir EBSPlungerFiles
cd EBSPlungerFiles
git clone https://github.com/waltmayfield/HilcorpPlungerLift
cd ../

docker create --gpus all -it --name testContainer --mount type=bind,source="$(pwd)"/EBSPlungerFiles,target=/EBSPlungerFiles tensorflow/tensorflow:latest-gpu

docker container start testContainer

docker exec testContainer pip install boto3
docker exec testContainer pip install pandas
docker exec testContainer pip install tqdm

docker container stop testContainer

