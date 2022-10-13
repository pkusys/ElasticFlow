"""
(MNMC) Multiple Nodes Multi-GPU Cards Training
    with DistributedDataParallel and torch.distributed.launch
Try to compare with [snsc.py, snmc_dp.py & mnmc_ddp_mp.py] and find out the differences.
"""

import math
import os
import time
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import argparse
import threading

import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils import stop_thread


LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class ResNetWorkload():
    CHECKPOINT_TEMPLATE = '/mnt/data1/resnet_{}.pth'
    DATA_PATH = "/mnt/data1/imagenet"

    def __init__(self, job_id, master_addr, master_port, rank, local_rank, world_size, batch_size, iterations, gpu_id) -> None:
        # job id
        self.job_id = job_id 
        # master address for process group
        self.master_addr = master_addr
        self.master_port = master_port
        # attribute to initialize process group
        self.rank = rank
        self.local_rank = local_rank        
        self.world_size = world_size
        # make sure global batch_size is fixed
        self.batch_size = batch_size
        # total iterations
        self.iterations = iterations
        self.gpu_id = gpu_id
        # identify the checkpoint path
        self.checkpoint_suffix = self.job_id
        self.checkpoint_path = ResNetWorkload.CHECKPOINT_TEMPLATE.format(self.checkpoint_suffix)
        self.logger_prefix = f"[job {self.job_id} | Resnet]"
        # training thread handle
        self.training_thread = None
        # data parallel process group
        self.group = None
        # received kill signal
        self.received_killed = False
        # indicate whether the model has been killed
        self.destructed = False
        # use to synchronize killing phase
        self.signal = None

    def init_dataset(self):
        # define dataloader
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        trainset = torchvision.datasets.ImageFolder(
            ResNetWorkload.DATA_PATH,transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

        # DistributedSampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            trainset,
            shuffle=True,
            rank=self.rank
        )
        self.train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
        )
        self.train_iter = iter(self.train_loader)
    
    @classmethod
    def load_from_checkpoint(cls, job_id, device):
        checkpoint_path = cls.CHECKPOINT_TEMPLATE.format(job_id)
        model = torchvision.models.resnet50().to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)) 
        return model

    def load_checkpoint(self):
        self.ori_model.load_state_dict(torch.load(self.checkpoint_path, map_location=self._get_device())) 
        return self.ori_model

    def init_model(self, from_scratch):
        # define network
        device = self._get_device()
        self.ori_model = self.ori_model.to(device)
        if not from_scratch:
            self.ori_model.load_state_dict(torch.load(self.checkpoint_path, map_location=device)) 

        self.signal = torch.Tensor([0]).to(self._get_device())

    def _get_device(self):
        return torch.device("cuda", self.gpu_id)

    def init_DDP_model(self, group):
        self.group = group
        self.ddp_model = DDP(self.ori_model, device_ids=[self.gpu_id], output_device=self.gpu_id, 
            process_group=self.group)

        # define loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.ddp_model.parameters(),
            lr=0.01 * 2,
            momentum=0.9,
            weight_decay=0.0001,
            nesterov=True,
        )

    def destruction(self):
        # !!!!!! synchronize all the cuda launches
        torch.cuda.synchronize(device=self._get_device())
        del self.optimizer
        del self.ori_model
        del self.ddp_model
        del self.train_loader
        # !!!!!! require this line to clear the cuda memory
        torch.cuda.set_device(device=self._get_device())
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device=self._get_device())


    def train_init(self, epochs=0, iteration_count=0):
        # start to train
        self.ddp_model.train()

        # track training process
        # (i) per training
        self.epochs = epochs
        self.iteration_count = iteration_count
        # (ii) per epoch
        self.idx = 0
        self.train_loss = 0
        self.correct = 0
        self.total = 0
        # (iii) use to checkpoint model with best accuracy
        self.best_acc = 0
        self.stable_reported = False
    
    #==============================================
    #        training utility function
    #==============================================
    def checkpoint(self):
        # only checkpoint one model copy on each node
        if self.local_rank % 8 != 0:
            return
        epoch_acc = 100.0 * self.correct / self.total
        #if self.best_acc < epoch_acc and self.local_rank == 0: # save one copy on each node
        if self.local_rank % 8 == 0: # save one copy on each node
            print(f"checkpoint the resnet model with acc {epoch_acc}%")
            self.best_acc = epoch_acc
            torch.save(self.ddp_model.module.state_dict(), ResNetWorkload.CHECKPOINT_TEMPLATE.format(self.job_id))
 

    def finish_train(self):
        self.checkpoint()
        self.destruction()
        self.destructed = True
    

    def train_iteration(self):
        try:
            inputs, targets = next(self.train_iter)
        except:
            # new epoch
            # self.checkpoint()
            self.train_iter = iter(self.train_loader)
            inputs, targets = next(self.train_iter)
            self.epochs += 1
            self.idx = 0
            self.train_loss = 0
            self.correct = 0
            self.total = 0

        self.idx += 1
        self.iteration_count += 1
        device = self._get_device()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = self.ddp_model(inputs)

        loss = self.criterion(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_loss += loss.item()
        self.total += targets.size(0)
        self.correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()

        if self.rank == 0 and ((self.idx + 1) % 25 == 0 or (self.idx + 1) == len(self.train_loader)):
            print(self.logger_prefix + 
                "[Epoch {}] [{}/{}] | loss: {:.3f} | acc: {:6.3f}%".format(
                    self.epochs,
                    self.idx,
                    len(self.train_loader),
                    self.train_loss / (self.idx + 1),
                    100.0 * self.correct / self.total,
                )
            )
        if self.received_killed == True:
            self.signal = torch.Tensor([1]).to(self._get_device())
        dist.all_reduce(self.signal, group=self.group, async_op=False)
        if self.signal.item() > 0:
            self.received_killed = True
            self.finish_train()
            return False
        
        return True

    #==========================================
    #            deprecated
    #==========================================
    # def start_train(self):
    #     self.training_thread = threading.Thread(
    #         target=self.train,
    #     )
    #     self.training_thread.start()
    
    # def is_started(self):
    #     return True
    #     return self.training_thread is not None

    # def is_finished(self):
    #     return False
    #     if self.training_thread is not None:
    #         return not self.training_thread.is_alive()
    #     else:
    #         print(self.logger_prefix + "training thread has not been started")
    #         return False

    # # def train(self, epoch=0, iteration_count=0):
    #     if self.received_killed:
    #         self.finish_train()
    #         return
    #     self.ddp_model.train()
    #     device = self._get_device()
    #     best_acc = 0.0
    #     self.signal = torch.Tensor([0]).to(self._get_device())
    #     stable_reported = False
    #     while True:
    #         train_loss = correct = total = 0
    #         # set sampler
    #         self.train_loader.sampler.set_epoch(epoch)

    #         for idx, (inputs, targets) in enumerate(self.train_loader):
    #             inputs, targets = inputs.to(device), targets.to(device)
    #             outputs = self.ddp_model(inputs)

    #             loss = self.criterion(outputs, targets)
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()

    #             train_loss += loss.item()
    #             total += targets.size(0)
    #             correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()

    #             if self.rank == 0 and ((idx + 1) % 5 == 0 or (idx + 1) == len(self.train_loader)):
    #                 print(self.logger_prefix + 
    #                     "[Epoch {}] [{}/{}] | loss: {:.3f} | acc: {:6.3f}%".format(
    #                     epoch,
    #                     idx,
    #                     len(self.train_loader),
    #                     train_loss / (idx + 1),
    #                     100.0 * correct / total,
    #                     )
    #                 )
        
    #             iteration_count += 1
    #             if iteration_count == self.iterations:
    #                 break
    #             if self.received_killed == True:
    #                 self.signal = torch.Tensor([1]).to(self._get_device())
    #             dist.all_reduce(self.signal, group=self.group, async_op=False)
    #             if self.signal.item() > 0:
    #                 self.received_killed = True
    #                 break
           
    #         # checkpoint the model in each epoch if the model is better
    #         epoch_acc = 100.0 * correct / total
    #         if best_acc < epoch_acc and self.local_rank == 0: # save one copy on each node
    #             print(f"checkpoint the model with acc {epoch_acc}%")
    #             best_acc = epoch_acc
    #             self.checkpoint()
    #         if self.received_killed == True:
    #             break
    #         if iteration_count == self.iterations:
    #             print(self.logger_prefix + "training finished!")
    #             break
    #         epoch += 1
        
    #     self.finish_train()
    