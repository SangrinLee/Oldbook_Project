# -*- coding: utf-8 -*-
import argparse
import torch
import torch.cuda as cuda
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import string
import pickle
import numpy as np
import bisect
import word_corpus_data as data
import re
import operator

#################################################
# Hyper-parameters
#################################################

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank Character-Level LSTM Model')
parser.add_argument('--lr', type=float, default=3, # original : 3
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--load_epochs', type=int, default=0,
                    help='load epoch')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=5,
                    help='sequence length')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
args = parser.parse_args()


#############################################
# Load data
#############################################
path = "word_data"
with open(path + '/corpus', 'rb') as handle:
    corpus = pickle.load(handle)

n_letters = len(corpus.dictionary)
print (n_letters)
# n_letters = 1000

# Load string of asterisked training and validation data
with open(path + '/train_data_array', 'rb') as handle:
    train_data_array = pickle.load(handle)
with open(path + '/val_data_array', 'rb') as handle:
    val_data_array = pickle.load(handle)

########################################################
# Pre-process training and validation data
########################################################
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[0: nbatch * bsz]
    # Evenly divide the data across the bsz batches.
    data = data.reshape(bsz, -1)
    return data

# for every batch you call this to get the batch of data and targets, in testing your only getting astericks for the characters you took out, 
# you have the index but the data is turned into 3D array, you have to map indices that you have, 
# you have to search for what astericks/how many are in the array, and then you go back using the indices of the astericks you already know
# to map that into indices of output array that you have 
def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, source.shape[1] - 1 - i) # -1 so that there's data for the target of the last time step
    data = source[:, i: i + seq_len] 
    source_target = source.astype(np.int64)
    target = source_target[:, i + 1: i + 1 + seq_len]

    data = torch.from_numpy(data)
    data = Variable(data, volatile=evaluation)    # Saves memory in purely inference mode

    target = torch.from_numpy(target)
    target = Variable(target, volatile=evaluation)
 
    target = target.contiguous().view(-1)
    return data.cuda(), target.cuda()

###############################################################################
# Build the model
###############################################################################
class MyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=3, layers=1, bias=True, batch_first=True, dropout=0.2):
        super(MyLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.direction = 1


        self.encoder = nn.Embedding(input_dim, 20) # added
        print ("# encoder", self.encoder)
        # for p in self.encoder.parameters():
            # print (p.size())
            # print (p.grad)

        self.lstm = nn.LSTM(20, hidden_dim, layers, bias, batch_first, dropout)
        print ("# lstm", self.lstm)
        # for p in self.lstm.parameters():
            # print (p.size())
            # print (p)
            # print (p.grad)
            

        self.hidden2category = nn.Linear(hidden_dim, input_dim)
        
        print ("# hidden2category", self.hidden2category)
        # for p in self.hidden2category.parameters():
        #     print (p.size())
        #     print (p)

        # exit()

    def init_hidden(self, length):
        return (torch.zeros(self.layers * self.direction, length, self.hidden_dim).cuda(),
                torch.zeros(self.layers * self.direction, length, self.hidden_dim).cuda())

    def forward(self, input_tensor, hidden): # input_tensor : 5*35*12043, hidden : 2*5*128
        print ("# INSIDE FORWARD #")

        # print ("#0", input_tensor.size())
        input_tensor = self.encoder(input_tensor)
        # print ("#1", input_tensor.size())
        output_tensor, hidden = self.lstm(input_tensor, hidden)
        # print ("#2", output_tensor.size())        
        # print ("#3", hidden[0].size(), hidden[1].size())
        output_tensor = output_tensor.contiguous().view(output_tensor.size(0)*output_tensor.size(1), output_tensor.size(2))
        # print ("#4", output_tensor.size())        
        output_tensor = self.hidden2category(output_tensor)
        # print ("#5", output_tensor.size())        
        # print ("#6", hidden[0].size(), hidden[1].size())
        return output_tensor, hidden



name = 'LSTM'

model = MyLSTM(n_letters)
model.cuda()

print (model)
# exit()
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
# optimizer = torch.optim.Adam([var1, var2], lr = 0.0001)
# optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)



criterion = nn.CrossEntropyLoss()
softmax = nn.Softmax()
NLL = nn.NLLLoss()

if args.cuda:
    criterion.cuda()
    softmax.cuda()
    NLL.cuda()

val_bsz = 5
train_data = batchify(train_data_array, args.batch_size)
val_data = batchify(val_data_array, val_bsz)

def train():
    # Turn on training mode which enables dropout.
    # Built-in function, has effect on dropout and batchnorm
    model.train()
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 1.2)

    batch_length = train_data_array.size // args.batch_size
    for batch, i in enumerate(range(1, train_data.shape[1] - 1, args.bptt)):
        # returns Variables
        data, targets = get_batch(train_data, i)
        # print (data)
        # exit()
        hidden = model.init_hidden(args.batch_size)
        
        # optimizer.zero_grad() # added
        model.zero_grad() # deleted

        output, hidden = model(data, hidden)
        loss = criterion(output, targets) 
        loss.backward()

        # optimizer.step() # added

        print ("# UPDATE PARAMETERS #")
        # torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            print (p.size())
            print (p.grad)
            print (p.grad.size())
            print (p.grad.data.size())
            p.data.add_(-lr, p.grad.data)   # (scalar multiplier, other tensor)
        exit()
        total_loss += loss.data
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:5.2f} |'.format(
                   epoch, batch, train_data.shape[1] // args.bptt, lr,
                   elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Uses training data to generate predictions, calculate loss based on validation/testing data
# Not using bptt
def evaluate():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(val_bsz)
    start_time = time.time()

    batch_length = val_data_array.size // val_bsz
    for batch, i in enumerate(range(1, val_data.shape[1] - 1, args.bptt)):
        # returns Variables
        data, targets = get_batch(val_data, i)

        hidden = model.init_hidden(val_bsz)

        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        total_loss += loss.data

        if batch % (args.log_interval // 20) == 0 and batch > 0:
            elapsed = time.time() - start_time
            print('| validation | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  .format(batch, val_data.shape[1] // args.bptt, lr,
                          elapsed * 1000 / (args.log_interval // 20)))
            start_time = time.time()
    return total_loss[0] / (batch_length / args.bptt)

# Loop over epochs.
lr = args.lr
best_val_loss = None

# Training Part
# At any point you can hit Ctrl + C to break out of training early.
arr1 = []
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate()
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} |'.format(
            epoch, (time.time() - epoch_start_time),
            val_loss))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(path + '/{}_Epoch{}_BatchSize{}_Dropout{}_LR{}_HiddenDim{}.pt'.format(
               name, args.epochs, args.batch_size, args.dropout, args.lr, args.nhid), 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        # else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            # lr /= 4.0
except KeyboardInterrupt:
   print('-' * 89)
   print('Exiting from training early')
