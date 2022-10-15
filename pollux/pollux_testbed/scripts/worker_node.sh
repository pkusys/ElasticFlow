#!/usr/bin/env bash
MASTER_IP=${1}
SSH_KEY=${2}

# config nfs
sudo apt install nfs-common --force-yes
sudo mount -t nfs $1:/mnt/data1 /mnt/data1

sudo apt install conntrack

MASTER_IP=${1}
SSH_KEY=${2}
sudo $(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ${SSH_KEY} $USER@${MASTER_IP} kubeadm token create --print-join-command)

conda env update -f ./environment.yaml # path

sudo helm install adaptdl adaptdl-sched --repo https://github.com/petuum/adaptdl/raw/helm-repo --namespace default --set docker-registry.enabled=true