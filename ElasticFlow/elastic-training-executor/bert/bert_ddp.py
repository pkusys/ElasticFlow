from transformers import BertForSequenceClassification, AdamW 

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, RandomSampler 
import numpy as np
import time
import datetime
import random
import os
import argparse
import math
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../scheduler'))


class BertWorkload():
    CHECKPOINT_TEMPLATE = '/mnt/data1/bert_{}_checkpoint'
    DATA_PATH = "/mnt/data1/data.pth"
    def __init__(self, job_id, master_addr, master_port, rank, local_rank, world_size, batch_size, iterations, gpu_id):
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
        self.checkpoint_path = BertWorkload.CHECKPOINT_TEMPLATE.format(self.checkpoint_suffix)
        self.logger_prefix = f"[job {self.job_id} | Bert]"
        # training thread handle
        # self.training_thread = None
        # data parallel process group
        self.group = None
        # received kill signal
        self.received_killed = False
        # indicate whether the model has been killed
        self.destructed = False
        # use to synchronize killing phase
        self.signal = None

    def _get_device(self):
        return torch.device("cuda", self.gpu_id)

    def init_dataset(self):
        # load training data
        data = torch.load(BertWorkload.DATA_PATH)
        input_ids = data["input_ids"]
        attention_masks = data["attention_masks"]
        labels = data["labels"]

        # create dataset
        trainset = TensorDataset(input_ids, attention_masks, labels)

        # create distributed sampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            trainset,
            shuffle=True,
        )

        # Create the DataLoaders for our training sets.
        self.train_loader = DataLoader(
                    trainset,  # The training samples.
                    sampler = train_sampler,
                    batch_size = self.batch_size # Trains with this batch size.
                )
        self.train_iter = iter(self.train_loader)  

    @classmethod
    def load_from_checkpoint(cls, job_id, device):
        checkpoint_path = cls.CHECKPOINT_TEMPLATE.format(job_id)
        model = BertForSequenceClassification.from_pretrained(checkpoint_path).to(device)
        return model

    def load_checkpoint(self):
        self.ori_model = BertForSequenceClassification.from_pretrained(self.checkpoint_path).to(self._get_device())
        return self.ori_model      

    def init_model(self, from_scratch):
        device = self._get_device()
        # Load BertForSequenceClassification, the pretrained BERT model with a single 
        # linear classification layer on top.
        #if os.path.exists(self.checkpoint_path):
        if not from_scratch:
            print("load pretrained model from checkpoint")
            self.ori_model = BertForSequenceClassification.from_pretrained(self.checkpoint_path).to(device)
        else:
            print("train model from scratch")
            """self.ori_model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 2, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
            ).to(device)"""
            self.ori_model.to(device)
        self.signal = torch.Tensor([0]).to(self._get_device())

    def init_DDP_model(self, group):
        self.group = group
        self.ddp_model = DDP(self.ori_model, device_ids=[self.gpu_id], output_device=self.gpu_id, process_group = self.group)

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
        # I believe the 'W' stands for 'Weight Decay fix"
        self.optimizer = AdamW(self.ddp_model.parameters(),
                        lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                        eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                        )

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

    #=================================================
    #       training utility function
    #=================================================
    def checkpoint(self):
        if self.local_rank % 8 != 0:
            return
        epoch_acc = 100.0 * self.correct / self.total
        #if self.best_acc < epoch_acc and self.local_rank == 0: # save one copy on each node
        if self.local_rank % 8 == 0: # save one copy on each node
            print(f"checkpoint the bert model with acc {epoch_acc}%")
            self.best_acc = epoch_acc
            self.ddp_model.module.save_pretrained(BertWorkload.CHECKPOINT_TEMPLATE.format(self.job_id))

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

    def finish_train(self):
        self.checkpoint()
        self.destruction()
        self.destructed = True

    def train_iteration(self):
        try:
            batch = next(self.train_iter)
        except:
            # new epoch
            # self.checkpoint()
            self.train_iter = iter(self.train_loader)
            batch = next(self.train_iter)
            self.epochs += 1
            self.idx = 0
            self.train_loss = 0
            self.correct = 0
            self.total = 0

        self.idx += 1
        self.iteration_count += 1
        device = self._get_device()
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        self.ddp_model.zero_grad()        

        # Specifically, we'll get the loss (because we provided labels) and the
        # "logits"--the model outputs prior to activation.
        result = self.ddp_model(b_input_ids, 
                       token_type_ids=None, 
                       attention_mask=b_input_mask, 
                       labels=b_labels,
                       return_dict=True)
        loss = result.loss
        logits = result.logits

        self.train_loss += loss.item()
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        self.optimizer.step()

        targets = b_labels.flatten()
        self.correct += torch.eq(logits.argmax(axis=1).flatten(), targets).sum().item()
        self.total += targets.size(0)

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
        #if (self.idx + 1) == len(self.train_loader):
        #    # sync
        #    acc = torch.Tensor([float(self.correct) / self.total])
        #    dist.all_reduce(acc, op=torch.distributed.ReduceOp.AVG, group=self.group, async_op=True)
            
        #    if acc.item() > 0.99:
        #        self.received_killed = True
        
        if self.received_killed == True:
            self.signal = torch.Tensor([1]).to(self._get_device())
        dist.all_reduce(self.signal, group=self.group, async_op=False)
        if self.signal.item() > 0:
            self.received_killed = True
            self.finish_train()
            return False
        
        return True