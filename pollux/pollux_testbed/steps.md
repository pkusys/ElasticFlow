trace：test0610.csv
dataset:
pretrained models and config:
/mnt/imagenet
/mnt/bert
/mnt/aclImdb
/mnt/LibriSpeech

/mnt/checkpoint/gpt2/



安装kubectl kubeadm kubelet
向集群添加节点：
https://blog.csdn.net/ikkyphoenix/article/details/119822169


install  (https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/)
CNI_VERSION="v0.8.2"
ARCH="amd64"
sudo mkdir -p /opt/cni/bin
curl -L "https://github.com/containernetworking/plugins/releases/download/${CNI_VERSION}/cni-plugins-linux-${ARCH}-${CNI_VERSION}.tgz" | sudo tar -C /opt/cni/bin -xz
DOWNLOAD_DIR=/usr/local/bin
sudo mkdir -p $DOWNLOAD_DIR
CRICTL_VERSION="v1.22.0"

ARCH="amd64"
curl -L "https://github.com/kubernetes-sigs/cri-tools/releases/download/${CRICTL_VERSION}/crictl-${CRICTL_VERSION}-linux-${ARCH}.tar.gz" | sudo tar -C $DOWNLOAD_DIR -xz
RELEASE="v1.21.0"
ARCH="amd64"
cd $DOWNLOAD_DIR
sudo curl -L --remote-name-all https://storage.googleapis.com/kubernetes-release/release/${RELEASE}/bin/linux/${ARCH}/{kubeadm,kubelet,kubectl}
sudo chmod +x {kubeadm,kubelet,kubectl}
RELEASE_VERSION="v0.4.0"
curl -sSL "https://raw.githubusercontent.com/kubernetes/release/${RELEASE_VERSION}/cmd/kubepkg/templates/latest/deb/kubelet/lib/systemd/system/kubelet.service" | sed "s:/usr/bin:${DOWNLOAD_DIR}:g" | sudo tee /etc/systemd/system/kubelet.service
sudo mkdir -p /etc/systemd/system/kubelet.service.d
curl -sSL "https://raw.githubusercontent.com/kubernetes/release/${RELEASE_VERSION}/cmd/kubepkg/templates/latest/deb/kubeadm/10-kubeadm.conf" | sed "s:/usr/bin:${DOWNLOAD_DIR}:g" | sudo tee /etc/systemd/system/kubelet.service.d/10-kubeadm.conf
systemctl enable --now kubelet


如果没有conda的话
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b -p ${HOME}/software/miniconda3
echo "export PATH=${HOME}/software/miniconda3/bin:\$PATH" >> ~/.bashrc
source ~/.bashrc
```

master:
sudo apt install conntrack
然后执行main.tf里的命令:
```
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
docker login -u ${var.docker_username} -p '${var.docker_password}'
kubectl create secret generic regcred --from-file=.dockerconfigjson=/home/ubuntu/.docker/config.json --type=kubernetes.io/dockerconfigjson
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh # install helm (https://helm.sh/docs/intro/install/)
helm repo add stable https://charts.helm.sh/stable --force-update
conda env update -f ~/adaptdl/benchmark/environment.yaml # path
```
`kubeadm token create --print-join-command`
然后去worker节点执行命令。


安装scheudler
sudo helm install adaptdl adaptdl-sched --repo https://github.com/petuum/adaptdl/raw/helm-repo --namespace default --set docker-registry.enabled=true

worker:
docker login -u ${var.docker_username} -p '${var.docker_password}'

all nodes:
sudo vim /etc/docker/daemon.json
改成https://github.com/NVIDIA/k8s-device-plugin#quick-start 这样
systemctl restart docker


跑实验：
python run_workload.py pollux test0617.csv --repository=gudiandian/pollux

cd adaptdl/sched
python setup.py install

https://github.com/Mellanox/k8s-rdma-sriov-dev-plugin.git
hca mode

https://www.cnblogs.com/zknublx/p/11010176.html
服务器端：
sudo vi /etc/exports
/mnt/checkpoint *(rw,sync,no_subtree_check,no_root_squash)
sudo service nfs-kernel-server restart

客户端
sudo apt install nfs-common (Y)
sudo mount -t nfs 10.5.0.7:/mnt/checkpoint /mnt/checkpoint

