import io
import os
import math
import torch
from torch._C import device
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import time
from sklearn.metrics import accuracy_score
from transformers import (GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../scheduler'))

class GPT2Workload():
    DATA_PATH = "/mnt/data1/aclImdb/train"
    CHECKPOINT_TEMPLATE = '/mnt/data1/gpt2_{}_checkpoint'
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
        self.checkpoint_path = GPT2Workload.CHECKPOINT_TEMPLATE.format(self.checkpoint_suffix)
        self.logger_prefix = f"[job {self.job_id} | GPT2]"
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

        #======================================
        #    workload specific attibute
        #======================================
        # Number of training epochs (authors on fine-tuning Bert recommend between 2 and 4).
        #epochs = 4

        # Pad or truncate text sequences to a specific length
        # if `None` it will use maximum sequence of word piece tokens allowed by model.
        self.max_length = 60

        # Name of transformers model - will use already pretrained model.
        # Path of transformer model - will load your own model from local disk.
        self.model_name_or_path = 'gpt2'
        # self.model_name_or_path = '/mnt/data1/gpt2_checkpoint' # in case of http error

        # Dictionary of labels and their id - this will be used to convert.
        # String labels to number ids.
        self.labels_ids = {'neg': 0, 'pos': 1}

        # How many labels are we using in training.
        # This is used to decide size of classification head.
        self.n_labels = len(self.labels_ids)

    def _get_device(self):
        return torch.device("cuda", self.gpu_id)

    def init_dataset(self):
        # Get model's tokenizer.
        #self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='gpt2')
        self.tokenizer = GPT2Tokenizer(vocab_file='/mnt/data1/gpt2/vocab.json', merges_file='/mnt/data1/gpt2/merges.txt')
        # default to left padding
        self.tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create data collator to encode text and labels into numbers.
        gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=self.tokenizer, 
                                                                labels_encoder=self.labels_ids, 
                                                                max_sequence_len=self.max_length)

        # Create pytorch dataset.
        train_dataset = MovieReviewsDataset(path=GPT2Workload.DATA_PATH,
                                    use_tokenizer=self.tokenizer)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            shuffle=True,
        )
    
        # Move pytorch dataset into dataloader.
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            collate_fn=gpt2_classificaiton_collator, 
            num_workers=1,
            pin_memory=True,
            sampler=train_sampler,
        )

        self.train_iter = iter(self.train_loader)

    @classmethod
    def load_from_checkpoint(cls, job_id, device):
        #tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='gpt2')
        tokenizer = GPT2Tokenizer(vocab_file='/mnt/data1/gpt2/vocab.json', merges_file='/mnt/data1/gpt2/merges.txt')
        #self.tokenizer.save_vocabulary('/mnt/data1/gpt2_checkpoint')
        #model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='gpt2', num_labels=2)
        model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='/mnt/data1/gpt2/', num_labels=2)
        checkpoint_path = cls.CHECKPOINT_TEMPLATE.format(job_id)
        model = GPT2ForSequenceClassification.from_pretrained(checkpoint_path, config=model_config)
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = model.config.eos_token_id
        model.to(device)
        return model

    def load_checkpoint(self):
        device = self._get_device()
        #model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=self.model_name_or_path, num_labels=self.n_labels)
        model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='/mnt/data1/gpt2/', num_labels=self.n_labels)
        self.ori_model = GPT2ForSequenceClassification.from_pretrained(self.checkpoint_path, config=model_config)
        self.ori_model.resize_token_embeddings(len(self.tokenizer))
        # fix model padding token id
        self.ori_model.config.pad_token_id = self.ori_model.config.eos_token_id
        self.ori_model.to(device)
        #self.ori_model.load_state_dict(torch.load(self.checkpoint_path, map_location=self._get_device())) 
        return self.ori_model
    
    def init_model(self, from_scratch):
        device = self._get_device()
        # Get model configuration.
        model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='/mnt/data1/gpt2/', num_labels=self.n_labels)
        #model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='gpt2', num_labels=self.n_labels)
        #if os.path.exists(self.checkpoint_path):
        if not from_scratch:
            self.ori_model = GPT2ForSequenceClassification.from_pretrained(self.checkpoint_path, config=model_config).to(device)
            # resize model embedding to match new tokenizer
            self.ori_model.resize_token_embeddings(len(self.tokenizer))
            # fix model padding token id
            self.ori_model.config.pad_token_id = self.ori_model.config.eos_token_id
        else:
            self.ori_model.to(device)
        #    self.ori_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=self.model_name_or_path, config=model_config).to(device)
        
        self.signal = torch.Tensor([0]).to(device)

    def init_DDP_model(self, group):
        self.group = group
        self.ddp_model = DDP(self.ori_model, device_ids=[self.gpu_id], output_device=self.gpu_id, process_group=self.group)
        # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
        # I believe the 'W' stands for 'Weight Decay fix"
        self.optimizer = AdamW(self.ddp_model.parameters(),
                        lr = 2e-5, # default is 5e-5, our notebook had 2e-5
                        eps = 1e-8 # default is 1e-8.
                        )
        # Create the learning rate scheduler.
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = self.iterations)

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
            print(f"checkpoint the gpt2 model with acc {epoch_acc}%")
            self.best_acc = epoch_acc
            self.ddp_model.module.save_pretrained(GPT2Workload.CHECKPOINT_TEMPLATE.format(self.job_id))

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

        # move batch to device
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        
        # Always clear any previously calculated gradients before performing a
        # backward pass.
        self.ddp_model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this a bert model function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = self.ddp_model(**batch)

        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple along with the logits. We will use logits
        # later to calculate training accuracy.
        loss, logits = outputs[:2]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        self.train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        self.optimizer.step()

        # Update the learning rate.
        self.scheduler.step()

        # Move logits and labels to CPU
        logits = logits.detach()

        # calculate correctness
        targets = batch['labels'].flatten()
        self.correct += torch.eq(logits.argmax(axis=-1).flatten(), targets).sum().item()
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
        
        if self.received_killed == True:
            self.signal = torch.Tensor([1]).to(self._get_device())
        dist.all_reduce(self.signal, group=self.group, async_op=False)
        if self.signal.item() > 0:
            self.received_killed = True
            self.finish_train()
            return False
        
        return True

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