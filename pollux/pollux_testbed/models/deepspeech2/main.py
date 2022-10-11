'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio

import math
import time
import sys
import numpy as np

import os
import argparse


import adaptdl
import adaptdl.torch

#from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from adaptdl.torch._metrics import report_train_metrics, report_valid_metrics


parser = argparse.ArgumentParser(description='PyTorch deepspeech Training')
parser.add_argument('--bs', default=64, type=int, help='batch size')
parser.add_argument('--lr', default=0.08, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
parser.add_argument('--model', default='ResNet18', type=str, help='model')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
#========================================================
#       Data preprocessing code                         #
#========================================================
# DATA_PATH = "./data"
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

def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets

def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]

def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)

def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer

print('==> Preparing data..') # todo
train_url="train-clean-100"
trainset = torchaudio.datasets.LIBRISPEECH("/mnt/", url=train_url, download=True)
train_audio_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
            torchaudio.transforms.TimeMasking(time_mask_param=100)
)
text_transform = TextTransform()


#trainset = torchvision.datasets.CIFAR10(root="/mnt", train=True, download=True, transform=transform_train)
trainloader = adaptdl.torch.AdaptiveDataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True,
    collate_fn=lambda x: data_processing(x, train_audio_transforms, text_transform, 'train'))
trainloader.autoscale_batch_size(256, local_bsz_bounds=(1, 64),
                                 gradient_accumulation=True)


# Model
print('==> Building model..')
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
net = SpeechRecognitionModel(
            hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
            hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        )


net = net.to(device)
#if device == 'cuda':
#    cudnn.benchmark = True

#criterion = nn.CrossEntropyLoss()
criterion = nn.CTCLoss(blank=28)
optimizer = optim.AdamW(net.parameters(), hparams['learning_rate'])
#lr_scheduler = ExponentialLR(optimizer, 0.0133 ** (1.0 / args.epochs))
print("will build nccl group")
adaptdl.torch.init_process_group("nccl")
print("nccl group build")
net = adaptdl.torch.AdaptiveDataParallel(net, optimizer)
samples = 0
# load from file
sample_file = os.getenv("ADAPTDL_CHECKPOINT_PATH", "/tmp") + "/samples.txt"
if os.path.exists(sample_file):
    with open(sample_file, 'r') as f:
        for line in f:
            samples = int(line.split('\n')[0])
            break

metric_achieved = False
if not os.path.exists(os.getenv("ADAPTDL_CHECKPOINT_PATH", "/tmp")):
    os.system("mkdir " + os.getenv("ADAPTDL_CHECKPOINT_PATH", "/tmp"))
target_samples = int(os.getenv("TARGET_SAMPLES")) #- samples
epochs = target_samples // (trainloader.batch_sampler.batch_size * torch.distributed.get_world_size() * len(trainloader))
if (target_samples % (trainloader.batch_sampler.batch_size * torch.distributed.get_world_size() * len(trainloader))) > 0:
    epochs += 1
print("epochs", epochs)
# Training
def train(epoch):
    global target_samples, samples, sample_file, metric_achieved
    print('\nEpoch: %d' % epoch)
    net.train()
    #stats = adaptdl.torch.Accumulator()
    total = 0
    print(len(trainloader), "iters per epoch")
    wers = []
    for batch in trainloader:
        spectrograms, labels, input_lengths, label_lengths = batch
        spectrograms, labels = spectrograms.to(device), labels.to(device)
        #print(len(inputs))
        #inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(spectrograms)
        outputs = F.log_softmax(outputs, dim=2)
        outputs = outputs.transpose(0, 1) # (time, batch, n_class)

        loss = criterion(outputs, labels, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()

        #stats["loss_sum"] += loss.item()


        _, predicted = outputs.max(1)
        #stats["total"] += labels.size(0)
        samples += int(labels.size(0)) * torch.distributed.get_world_size()
        total += 1
        #stats["correct"] += predicted.eq(targets).sum().item()
        decoded_preds, decoded_targets = GreedyDecoder(outputs.transpose(0, 1), labels, label_lengths)
        for j in range(len(decoded_preds)):
            wers.append(wer(decoded_targets[j], decoded_preds[j]))

        #trainloader.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Data")
        #net.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Model")
        if torch.distributed.get_rank() == 0:
            #remove file
            if os.path.exists(sample_file):
                os.system("rm " + sample_file)
            with open(sample_file, 'a') as f:
                f.write(str(samples) + "\n")
        # write file
        # get checkpoint path. write file from start. 
        print(samples, "samples")
        if samples >= target_samples:
            print("reach target!")
            break

    """with stats.synchronized():
        stats["loss_avg"] = stats["loss_sum"] / total
        stats["avg_wer"] = sum(wers)/len(wers)
        #stats["accuracy"] = stats["correct"] / stats["total"]
        writer.add_scalar("Loss/Train", stats["loss_avg"], epoch)
        #writer.add_scalar("Accuracy/Train", stats["accuracy"], epoch)
        report_train_metrics(epoch, stats["loss_avg"], wer=stats["avg_wer"])
        print("Train:", stats)"""
    """if torch.distributed.get_rank() == 0:
        if stats._state.results["avg_wer"] <= 0.25:
            metric_achieved = True
            #remove file
            if os.path.exists(sample_file):
                os.system("rm " + sample_file)
            with open(sample_file, 'a') as f:
                f.write(str(target_samples) + "\n")"""



with SummaryWriter(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp")) as writer:
    for epoch in adaptdl.torch.remaining_epochs_until(epochs):
        train(epoch)
        """if not metric_achieved:
            if os.path.exists(sample_file):
                with open(sample_file, 'r') as f:
                    for line in f:
                        samples = int(line.split('\n')[0])
                        break"""
        metric_achieved =  (samples >= target_samples)
        if metric_achieved:
            break
        #valid(epoch)
        #lr_scheduler.step()
