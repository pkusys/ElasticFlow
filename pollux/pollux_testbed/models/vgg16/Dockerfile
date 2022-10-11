FROM nvcr.io/nvidia/pytorch:21.10-py3
WORKDIR /root

COPY adaptdl/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY benchmark/models/vgg16/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY adaptdl adaptdl
ENV PYTHONPATH=/root/adaptdl:$PYTHONPATH

COPY benchmark/models/vgg16 vgg16

COPY benchmark/topo.xml topo.xml
ENV NCCL_IB_PCI_RELAXED_ORDERING=1
ENV NCCL_SOCKET_IFNAME=eth0
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV NCCL_NET_GDR_LEVEL=5
ENV LD_PRELOAD=/opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so
ENV LD_LIBRARY_PATH=/opt/hpcx/nccl_rdma_sharp_plugin/lib:$LD_LIBRARY_PATH 
ENV NCCL_TOPO_FILE=/root/topo.xml
ENV PYTHONPATH=/root/vgg16:$PYTHONPATH
WORKDIR /root/vgg16

ENV PYTHONUNBUFFERED=true
