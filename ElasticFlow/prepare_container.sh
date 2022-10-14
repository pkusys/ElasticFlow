#!/usr/bin/env bash

sudo mkfs.ext4 /dev/nvme1n1 
sudo mkdir /mnt/data1 
sudo mount /dev/nvme1n1 /mnt/data1 
sudo chmod 777 /mnt/data1