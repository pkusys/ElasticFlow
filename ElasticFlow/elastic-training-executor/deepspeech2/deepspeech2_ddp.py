"""
This script is adapted from https://colab.research.google.com/drive/1IPpwx4rX32rqHKpLz7dc8sOKspUa-YKO#scrollTo=ydkqGeOwnPGY
"""
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse
import math
import time
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../scheduler'))
#from runtime.rpc import trainer_client

#========================================================
#       Data preprocessing code                         #
#========================================================
class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')


def data_processing(data, train_audio_transforms, text_transform, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif data_type == 'valid':
            # spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            # do not need valid dataset
            pass 
        else:
            raise Exception('data_type should be train or valid')
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths


#======================================================
#       deepspeech2 model                             #
#======================================================
class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):
    
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x

#====================================================
#           workload
#====================================================
class DeepSpeechWorkload():
    CHECKPOINT_TEMPLATE = '/mnt/data1/deepspeech2_{}.pth'
    DATA_PATH = "/mnt/data1"
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
        self.checkpoint_path = DeepSpeechWorkload.CHECKPOINT_TEMPLATE.format(self.checkpoint_suffix)
        self.logger_prefix = f"[job {self.job_id} | DeepSpeech2]"
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
        # process dataset
        if not os.path.isdir(DeepSpeechWorkload.DATA_PATH):
            os.makedirs(DeepSpeechWorkload.DATA_PATH)
        train_url="train-clean-100"
        train_dataset = torchaudio.datasets.LIBRISPEECH(DeepSpeechWorkload.DATA_PATH, url=train_url, download=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            shuffle=True,
        )
        train_audio_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
            torchaudio.transforms.TimeMasking(time_mask_param=100)
        )
        text_transform = TextTransform()
        self.train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda x: data_processing(x, train_audio_transforms, text_transform, 'train'),
            num_workers=1, # prevent cuda c10::Error
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True
        )
        self.train_iter = iter(self.train_loader)

    @classmethod
    def load_from_checkpoint(cls, job_id, device):
        checkpoint_path = cls.CHECKPOINT_TEMPLATE.format(job_id)
        hparams = {
            "n_cnn_layers": 3,
            "n_rnn_layers": 5,
            "rnn_dim": 512,
            "n_class": 29,
            "n_feats": 128,
            "stride":2,
            "dropout": 0.1,
            "learning_rate": 5e-4,
        }
        model = SpeechRecognitionModel(
            hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
            hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        ).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)) 
        return model

    def load_checkpoint(self):
        self.ori_model.load_state_dict(torch.load(self.checkpoint_path, map_location=self._get_device())) 
        return self.ori_model

        
    def init_model(self, from_scratch):
        device = self._get_device()
        self.hparams = {
            "n_cnn_layers": 3,
            "n_rnn_layers": 5,
            "rnn_dim": 512,
            "n_class": 29,
            "n_feats": 128,
            "stride":2,
            "dropout": 0.1,
            "learning_rate": 5e-4,
        }
        self.ori_model.to(device)

        if not from_scratch:
            self.ori_model.load_state_dict(torch.load(self.checkpoint_path, map_location=device))  
        self.signal = torch.Tensor([0]).to(self._get_device())

    def init_DDP_model(self, group):
        self.group = group
        self.ddp_model = DDP(self.ori_model, device_ids=[self.gpu_id], output_device=self.gpu_id, process_group=self.group)
        device = self._get_device()
        self.optimizer = optim.AdamW(self.ddp_model.parameters(), self.hparams['learning_rate'])
        self.criterion = nn.CTCLoss(blank=28).to(device)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.hparams['learning_rate'], 
                                                steps_per_epoch=int(len(self.train_loader)),
                                                epochs=max(1, math.ceil(self.iterations / len(self.train_loader))),
                                                anneal_strategy='linear')

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
        if self.local_rank % 8 == 0:
            torch.save(self.ddp_model.module.state_dict(), DeepSpeechWorkload.CHECKPOINT_TEMPLATE.format(self.job_id))
            print("checkpoint the deepspeech2 model")

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

        self.idx += 1
        self.iteration_count += 1
        device = self._get_device()

        spectrograms, labels, input_lengths, label_lengths = batch
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        self.optimizer.zero_grad()

        output = self.ddp_model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)

        loss = self.criterion(output, labels, input_lengths, label_lengths)
        loss.backward()
        self.train_loss += loss.item()
        self.optimizer.step()
        self.scheduler.step()
 
        if self.rank == 0 and ((self.idx + 1) % 25 == 0 or (self.idx + 1) == len(self.train_loader)):
            print(self.logger_prefix + 
                "[Epoch {}] [{}/{}] | loss: {:.3f}".format(
                    self.epochs,
                    self.idx,
                    len(self.train_loader),
                    self.train_loss / (self.idx + 1),
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