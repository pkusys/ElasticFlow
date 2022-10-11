'''Train BERT with PyTorch.'''
import io
import os
import math
import torch
from torch._C import device
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, RandomSampler 
#import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from sklearn.metrics import accuracy_score
from transformers import BertForSequenceClassification, AdamW 

import adaptdl
import adaptdl.torch

from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from adaptdl.torch._metrics import report_train_metrics, report_valid_metrics
import time


parser = argparse.ArgumentParser(description='PyTorch BERT Training')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
data = torch.load("/mnt/bert/data.pth")
input_ids = data["input_ids"]
attention_masks = data["attention_masks"]
labels = data["labels"]

# create dataset
trainset = TensorDataset(input_ids, attention_masks, labels)
#trainset = Dataset(trainset)

trainloader = adaptdl.torch.AdaptiveDataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
trainloader.autoscale_batch_size(256, local_bsz_bounds=(1, 256),
                                 gradient_accumulation=True)

# Model
print('==> Building model..')
model = BertForSequenceClassification.from_pretrained(
    "bert-large-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
) # todo: load from pretrained
#model = BertForSequenceClassification.from_pretrained("/mnt/checkpoint/bert")
model = model.to(device)

#if device == 'cuda':
#    cudnn.benchmark = True

#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD([{"params": [param]} for param in net.parameters()],
optimizer = AdamW(model.parameters(),
                        lr = 2e-5, # default is 5e-5, our notebook had 2e-5
                        eps = 1e-8 # default is 1e-8.
                        )
#lr_scheduler = get_linear_schedule_with_warmup(optimizer, 
#    num_warmup_steps = 0, # Default value in run_glue.py
#    #num_training_steps = self.iterations
#    )


adaptdl.torch.init_process_group("nccl")
#model = adaptdl.torch.AdaptiveDataParallel(model, optimizer, lr_scheduler)
model = adaptdl.torch.AdaptiveDataParallel(model, optimizer)
model.zero_grad()
samples = 0
# load from file
if not os.path.exists(os.getenv("ADAPTDL_CHECKPOINT_PATH", "/tmp")):
    os.system("mkdir " + os.getenv("ADAPTDL_CHECKPOINT_PATH", "/tmp"))
sample_file = os.getenv("ADAPTDL_CHECKPOINT_PATH", "/tmp") + "/samples.txt"
if os.path.exists(sample_file):
    with open(sample_file, 'r') as f:
        for line in f:
            samples = int(line.split('\n')[0])
            break

target_samples = int(os.getenv("TARGET_SAMPLES")) #- samples
epochs = target_samples // (trainloader.batch_sampler.batch_size * torch.distributed.get_world_size() * len(trainloader))
if (target_samples % (trainloader.batch_sampler.batch_size * torch.distributed.get_world_size() * len(trainloader))) > 0:
    epochs += 1
# Training
metric_achieved = False
def train(epoch):
    global target_samples, samples, sample_file, metric_achieved 
    start_time = time.time()
    print('\nEpoch: %d' % epoch)
    model.train()
    #stats = adaptdl.torch.Accumulator()
    for batch in trainloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, 
          labels=b_labels, return_dict=True)
        loss = result.loss
        logits = result.logits

        targets = b_labels.flatten()
        
        loss.backward()
        optimizer.step()

        #stats["loss_sum"] += loss.item() * b_labels.size(0)
        #stats["correct"] +=  torch.eq(logits.argmax(axis=1).flatten(), targets).sum().item()
        #_, predicted = outputs.max(1)
        #stats["total"] += targets.size(0)
        samples += int(targets.size(0)) * torch.distributed.get_world_size()
        if torch.distributed.get_rank() == 0:
            #remove file
            if os.path.exists(sample_file):
                os.system("rm " + sample_file)
            with open(sample_file, 'a') as f:
                f.write(str(samples) + "\n")
        #stats["correct"] += predicted.eq(batch['labels']).sum().item()

        trainloader.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Data")
        model.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Model")
        model.zero_grad()
        if samples >= target_samples:
            break

    """with stats.synchronized():
        stats["loss_avg"] = stats["loss_sum"] / stats["total"]
        stats["accuracy"] = stats["correct"] / stats["total"]
        writer.add_scalar("Loss/Train", stats["loss_avg"], epoch)
        writer.add_scalar("Accuracy/Train", stats["accuracy"], epoch)
        report_train_metrics(epoch, stats["loss_avg"], accuracy=stats["accuracy"])
        print("Train:", stats)"""
    #model.module.save_pretrained("/mnt/checkpoint/bert")
    #print("tpt: ", len(trainloader) / (time.time() - start_time))

    #if stats._state.results["accuracy"] > 0.99:
    #    metric_achieved = True
    """if torch.distributed.get_rank() == 0:
        if stats._state.results["accuracy"] >= 0.99:
            metric_achieved = True
            #remove file
            if os.path.exists(sample_file):
                os.system("rm " + sample_file)
            with open(sample_file, 'a') as f:
                f.write(str(target_samples) + "\n")"""

#bs=64: 14epoch
with SummaryWriter(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp")) as writer:
    for epoch in adaptdl.torch.remaining_epochs_until(epochs):
        train(epoch)
        #if metric_achieved:
        #    break
        #torch.save(model.module.state_dict(), "/mnt/checkpoint/test")
        #break
        #model.module.save_pretrained("/mnt/checkpoint/bert/")
        #break
        #valid(epoch)
        #lr_scheduler.step()
        """if not metric_achieved:
            if os.path.exists(sample_file):
                with open(sample_file, 'r') as f:
                    for line in f:
                        samples = int(line.split('\n')[0])
                        break"""
        metric_achieved =  (samples >= target_samples)
        if metric_achieved:
            break
