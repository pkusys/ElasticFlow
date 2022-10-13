'''Train GPT2 with PyTorch.'''
import io
import os
import math
import torch
from torch._C import device
from torch.utils.data import Dataset, DataLoader
#import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from sklearn.metrics import accuracy_score
from transformers import (GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)

import adaptdl
import adaptdl.torch

from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from adaptdl.torch._metrics import report_train_metrics, report_valid_metrics


parser = argparse.ArgumentParser(description='PyTorch GPT2 Training')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
args = parser.parse_args()

#============================================================
#                data processing
#============================================================
class MovieReviewsDataset(Dataset):
  r"""PyTorch Dataset class for loading data.

  This is where the data parsing happens.

  This class is built with reusability in mind: it can be used as is as.

  Arguments:

    path (:obj:`str`):
        Path to the data partition.

  """

  def __init__(self, path, use_tokenizer):

    # Check if path exists.
    if not os.path.isdir(path):
      # Raise error if path is invalid.
      raise ValueError('Invalid `path` variable! Needs to be a directory')
    self.texts = []
    self.labels = []
    # Since the labels are defined by folders with data we loop 
    # through each label.
    for label in ['pos', 'neg']:
      sentiment_path = os.path.join(path, label)

      # Get all files from path.
      files_names = os.listdir(sentiment_path)#[:10] # Sample for debugging.
      # Go through each file and read its content.
      for file_name in files_names:
        file_path = os.path.join(sentiment_path, file_name)

        # Read content.
        content = io.open(file_path, mode='r', encoding='utf-8').read()
        # Fix any unicode issues.
        # content = fix_text(content)
        # Save content.
        self.texts.append(content)
        # Save encode labels.
        self.labels.append(label)

    # Number of exmaples.
    self.n_examples = len(self.labels)
    

    return

  def __len__(self):
    r"""When used `len` return the number of examples.

    """
    
    return self.n_examples

  def __getitem__(self, item):
    r"""Given an index return an example from the position.
    
    Arguments:

      item (:obj:`int`):
          Index position to pick an example to return.

    Returns:
      :obj:`Dict[str, str]`: Dictionary of inputs that contain text and 
      asociated labels.

    """

    return {'text':self.texts[item],
            'label':self.labels[item]}

class Gpt2ClassificationCollator(object):
    r"""
    Data Collator used for GPT2 in a classificaiton rask. 
    
    It uses a given tokenizer and label encoder to convert any text and labels to numbers that 
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to 
          labels names and Values map to number associated to those labels.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):

        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        # Label encoder used inside the class.
        self.labels_encoder = labels_encoder

        return

    def __call__(self, sequences):
        r"""
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this 
        class as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """

        # Get all texts from sequences list.
        texts = [sequence['text'] for sequence in sequences]
        # Get all labels from sequences list.
        labels = [sequence['label'] for sequence in sequences]
        # Encode all labels using label encoder.
        labels = [self.labels_encoder[label] for label in labels]
        # Call tokenizer on all texts to convert into tensors of numbers with 
        # appropriate padding.
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels':torch.tensor(labels)})

        return inputs

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
#tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='gpt2')
tokenizer = GPT2Tokenizer(vocab_file='/mnt/data1/gpt2/vocab.json', merges_file='/mnt/data1/gpt2/merges.txt')
# todo: use fixed tokenizer
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer, 
                                                                labels_encoder={'neg': 0, 'pos': 1}, 
                                                                max_sequence_len=64)
trainset = MovieReviewsDataset(path="/mnt/data1/aclImdb/train",
                                    use_tokenizer=tokenizer)


trainloader = adaptdl.torch.AdaptiveDataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True, collate_fn=gpt2_classificaiton_collator)
trainloader.autoscale_batch_size(256, local_bsz_bounds=(1, 256),
                                 gradient_accumulation=True)

# Model
print('==> Building model..')

model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='/mnt/data1/gpt2/', num_labels=2)
# if the model exist, load from checkpoint
if os.path.exists(os.getenv("ADAPTDL_CHECKPOINT_PATH", "/tmp")+"/pytorch_model.bin"):
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.getenv("ADAPTDL_CHECKPOINT_PATH", "/tmp"), config=model_config)
else:
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path='/mnt/data1/gpt2/', config=model_config)
# resize model embedding to match new tokenizer
model.resize_token_embeddings(len(tokenizer))
# fix model padding token id
model.config.pad_token_id = model.config.eos_token_id
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
#model = adaptdl.torch.AdaptiveDataParallel(model, optimizer, model_type="gpt2")
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
metric_achieved = False
target_samples = int(os.getenv("TARGET_SAMPLES")) #- samples
epochs = target_samples // (trainloader.batch_sampler.batch_size * torch.distributed.get_world_size() * len(trainloader))
if (target_samples % (trainloader.batch_sampler.batch_size * torch.distributed.get_world_size() * len(trainloader))) > 0:
    epochs += 1
# Training
def train(epoch):
    global sample_file, metric_achieved, samples
    print('\nEpoch: %d' % epoch)
    model.train()
    #stats = adaptdl.torch.Accumulator()
    for batch in trainloader:
        true_labels = batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}

        outputs = model(**batch)
        loss, logits = outputs[:2]
        logits = logits.detach().cpu().numpy()
        predict_content = logits.argmax(axis=-1).flatten().tolist()
        
        loss.backward()
        optimizer.step()

        #stats["loss_sum"] += loss.item() * batch['labels'].size(0)
        train_acc = accuracy_score(true_labels, predict_content)
        #stats["acc"] += train_acc
        #_, predicted = outputs.max(1)
        #stats["total"] += 1
        samples += len(true_labels) * torch.distributed.get_world_size()
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
        stats["accuracy"] = stats["acc"] / stats["total"]
        writer.add_scalar("Loss/Train", stats["loss_avg"], epoch)
        writer.add_scalar("Accuracy/Train", stats["accuracy"], epoch)
        report_train_metrics(epoch, stats["loss_avg"], accuracy=stats["accuracy"])
        print("Train:", stats)"""
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


with SummaryWriter(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp")) as writer:
    for epoch in adaptdl.torch.remaining_epochs_until(epochs):
        train(epoch)
        #torch.save(model.module.state_dict(), "/mnt/checkpoint/test")
        #model.module.save_pretrained("/mnt/checkpoint/gpt2/")
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
