#!/usr/bin/env bash

# config nfs
#/mnt/data1 *(rw,sync,no_subtree_check,no_root_squash)
sudo service nfs-kernel-server restart
sudo apt install conntrack

sudo snap install yq --channel=v3/stable
sudo kubeadm init --pod-network-cidr=192.168.0.0/16
mkdir -p ~/.kube
sudo cp /etc/kubernetes/admin.conf ~/.kube/config
sudo chown -f -R $USER ~/.kube
kubectl apply -f https://docs.projectcalico.org/v3.23/manifests/calico.yaml # weave
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/1.0.0-beta5/nvidia-device-plugin.yml
kubectl apply -f https://raw.githubusercontent.com/rook/rook/v1.3.1/cluster/examples/kubernetes/ceph/common.yaml
kubectl apply -f https://raw.githubusercontent.com/rook/rook/v1.3.1/cluster/examples/kubernetes/ceph/operator.yaml
curl -s https://raw.githubusercontent.com/rook/rook/v1.3.1/cluster/examples/kubernetes/ceph/cluster.yaml | /snap/bin/yq w - spec.storage.deviceFilter nvme0n1p2 | kubectl apply -f -
kubectl apply -f https://raw.githubusercontent.com/rook/rook/v1.3.1/cluster/examples/kubernetes/ceph/filesystem.yaml
kubectl apply -f https://raw.githubusercontent.com/rook/rook/v1.3.1/cluster/examples/kubernetes/ceph/csi/cephfs/storageclass.yaml
kubectl create secret generic regcred --from-file=.dockerconfigjson=/home/$USER/.docker/config.json --type=kubernetes.io/dockerconfigjson
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh # install helm (https://helm.sh/docs/intro/install/)
helm repo add stable https://charts.helm.sh/stable --force-update
conda env update -f ./environment.yaml # path

kubeadm token create --print-join-command

sudo helm install adaptdl adaptdl-sched --repo https://github.com/petuum/adaptdl/raw/helm-repo --namespace default --set docker-registry.enabled=true
