import threading
import time
import inspect
import ctypes
import torchvision
from transformers import (GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification,
    BertForSequenceClassification)
import torch.nn as nn
import torch.nn.functional as F
 
def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")
 
def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)

# only for test 
def print_time():
    while 2:
         print(111111111111)
         print(222222222222)
         print(333333333333)
         print(444444444444)
         print(555555555555)
         print(666666666666)

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

def model_list():
    resnet_model = torchvision.models.resnet50()
    resnet_cifar10_model = torchvision.models.resnet50()
    resnet_cifar10_model.fc = nn.Linear(resnet_cifar10_model.fc.in_features, 10)
    vgg_model = torchvision.models.vgg16()
    inception_model = torchvision.models.inception_v3(init_weights=False, aux_logits=False)
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='gpt2', num_labels=2)
    #model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='/mnt/data1/gpt2/', num_labels=2)
    #gpt2_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path='/mnt/data1/gpt2/', config=model_config)
    gpt2_model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path='gpt2', config=model_config)
    # resize model embedding to match new tokenizer
    #tokenizer = GPT2Tokenizer(vocab_file='/mnt/data1/gpt2/vocab.json', merges_file='/mnt/data1/gpt2/merges.txt')
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='gpt2')
    gpt2_model.resize_token_embeddings(len(tokenizer))
    # fix model padding token id
    gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id
    bert_model = BertForSequenceClassification.from_pretrained(
        "bert-large-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    #bert_model = BertForSequenceClassification.from_pretrained("/mnt/data1/bert")
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
    deepspeech_model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
    )

    model_list = {
        "resnet50" : resnet_model,
        "resnet50_cifar10": resnet_cifar10_model,
        "vgg16" : vgg_model,
        "inception3" : inception_model,
        "gpt2" : gpt2_model,
        "bert" : bert_model,
        "deepspeech2" : deepspeech_model,
    }
    return model_list
 
 
if __name__ == "__main__":
    t = threading.Thread(target=print_time)
    t.start()
 
    stop_thread(t)
    print("stoped")
    print(t.is_alive())
    time.sleep(3)

