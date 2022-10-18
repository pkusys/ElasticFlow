#!/usr/bin/env bash

sudo mkfs.ext4 /dev/nvme1n1 
sudo mkdir /mnt/data1 
sudo mount /dev/nvme1n1 /mnt/data1 
sudo chmod 777 /mnt/data1

sudo nvidia-smi -pm 1

sudo docker run -it -d --name=ddl --privileged --net=host --ipc=host --gpus=all -v /mnt:/mnt   gudiandian/elasticflow:v1.3
sudo docker exec -it ddl bash
