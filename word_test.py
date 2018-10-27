# -*- coding: utf-8 -*-
import argparse
import torch
import torch.cuda as cuda
from torch.autograd import Variable
from word_lstm_model import *
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
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=3, # original : 3
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--load_epochs', type=int, default=0,
                    help='load epoch')
parser.add_argument('--epochs', type=int, default=15,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=5, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='use Bi-LSTM')
parser.add_argument('--serialize', action='store_true', default=False, #False,
                    help='continue training a stored model')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
args = parser.parse_args()

def progress(count, total, status=''):
    bar_len = 30
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

#############################################
# Load data
#############################################
path = "word_data"
with open(path + '/corpus', 'rb') as handle:
    corpus = pickle.load(handle)

# Variables to fix the ordering problem
temp1 = []
temp2 = []
temp3 = {}
temp4 = {}
averages=[]

# Load testing word and each index
with open(path + '/test_data', 'rb') as handle:
    test_data = pickle.load(handle)

# Load dictionaries of actual asterisked characters
with open('word_data/actual_test_data', 'rb') as f:
    test_data_list=pickle.load(f)
# print ("----- test_data_list -----")
# print (test_data_list)
# print (len(test_data_list))

with open('freq_data/user_input', 'rb') as f:
    year=pickle.load(f)
year1 = year[0]

year_round=math.floor(int(year1)/10)
with open('word_ptb_models/dict_array_'+str(year_round)+'1','rb') as f:
    corpus_load=pickle.load(f)

n_letters = len(corpus_load.dictionary)
n_categories = len(corpus_load.dictionary)
print (n_letters)

# Load string of asterisked training and validation data
with open(path + '/train_data_array', 'rb') as handle:
    train_data_array = pickle.load(handle)
with open(path + '/val_data_array', 'rb') as handle:
    val_data_array = pickle.load(handle)

# all_data_array = np.append(val_data_array, train_data_array)

with open('word_data/all_data_array', 'rb') as handle:
    all_data_array = pickle.load(handle)

test_data.sort(key=lambda x: x[0], reverse=False)
test_target_index = []
test_target_tensor = torch.LongTensor(len(test_data)).zero_()

for i, char in enumerate(test_data):
    test_target_index.append(char[0])
    test_target_tensor[i] = int(char[1])

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

# Wraps hidden states in new Variables, to detach them from their history.
def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

# for every batch you call this to get the batch of data and targets, in testing your only getting astericks for the characters you took out, 
# you have the index but the data is turned into 3D array, you have to map indices that you have, 
# you have to search for what astericks/how many are in the array, and then you go back using the indices of the astericks you already know
# to map that into indices of output array that you have 
def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, source.shape[1] - 1 - i) # -1 so that there's data for the target of the last time step
    data = source[:, i: i + seq_len] 
    source_target = source.astype(np.int64)
    target = source_target[:, i + 1: i + 1 + seq_len]

    # initialize train_data_tensor, test_data_tensor
    data_embedding = np.zeros((data.shape[0], data.shape[1], n_letters), dtype = np.float32)

    # convert 2D numpy array to 3D numpy embedding
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            data_embedding[i][j][data[i][j]] = 1

    # create tensor variable
    data_embedding = torch.from_numpy(data_embedding)
    data_embedding = Variable(data_embedding, volatile=evaluation)    # Saves memory in purely inference mode

    target = torch.from_numpy(target)
    target = Variable(target, volatile=evaluation)
    if args.bidirectional:
        # r_target of length seq_len - 1
        r_source_target = np.flip(source_target[:, i - 1: i - 1 + seq_len].cpu().numpy(), 1).copy()
        target = torch.cat((Variable(source_target[:, i + 1: i + 1 + seq_len].contiguous().view(-1)),
                            Variable(torch.from_numpy(r_source_target).cuda().contiguous().view(-1))), 0)
    else:
        target = target.contiguous().view(-1)
    if args.cuda:
        return data_embedding.cuda(), target.cuda()
    else:
        return data_embedding, target

def embed(data_array, bsz):
    # convert 1D array to 2D
    data_array = batchify(data_array, bsz)

    # initialize train_data_tensor, test_data_tensor
    data_embedding = np.zeros((data_array.shape[0], data_array.shape[1], n_letters), dtype = np.float32)

    # convert 2D numpy array to np.int64
    data_array = data_array.astype(np.int64)

    # convert 2D numpy array to 3D numpy embedding
    for i in range(0, data_array.shape[0]):
        for j in range(0, data_array.shape[1]):
            data_embedding[i][j][data_array[i][j]] = 1

    # convert 2D numpy array to 2D target tensor
    return data_embedding, data_array

def find_ge(a, x):
    i = bisect.bisect_left(a, x)
    return i

def find_le(a, x):
    i = bisect.bisect_right(a, x)
    return i - 1

###############################################################################
# Settings for the model
###############################################################################

# Load the model based on the user's input
with open('word_ptb_models/' + str(year_round) + '1.pt', 'rb') as f:
    model = torch.load(f)

if args.bidirectional:
    name = 'Bi-LSTM'
else:
    name = 'LSTM'

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
softmax = nn.Softmax()
NLL = nn.NLLLoss()

if args.cuda:
    criterion.cuda()
    softmax.cuda()
    NLL.cuda()

val_bsz = 5
all_data = batchify(all_data_array, args.batch_size)
train_data = batchify(train_data_array, args.batch_size)
val_data = batchify(val_data_array, val_bsz)

def test():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idx = 0
    total_loss = 0
    correct_count = 0
    not_in_fixed_list_count = 0
    total_count = 0
    high_correct = 0
    high_total = 0
    last_forward = None
    batch_length = all_data_array.size // args.batch_size
    hidden = model.init_hidden(args.batch_size)
    start_time = time.time()
    progress_counter=0

    for batch, i in enumerate(range(1, all_data.shape[1] - 1, args.bptt)):
        # returns Variables
        data, _ = get_batch(all_data, i)

        hidden = model.init_hidden(args.batch_size)
        output, hidden = model(data, hidden)
        output_select = Variable(torch.FloatTensor(output.size(0), n_categories).zero_())
        target_select = Variable(torch.LongTensor(output.size(0)).zero_())
        index_select = []

        count = 0
        bptt = min(batch_length - i, args.bptt)

        for batch_i in range(0, args.batch_size):
            start = find_ge(test_target_index, i + 1 + batch_i * batch_length)
            end = find_le(test_target_index, i + batch_i * batch_length + bptt) + 1
            for ii in range(start, end):
                target = test_target_index[ii]
                temp1.append(target)

                if target - batch_i * (batch_length - bptt) - i - 1 >= len(output):
                    output_select[count] = output[len(output) - 1]
                else:
                    output_select[count] = output[target - batch_i * (batch_length - bptt) - i - 1]
                target_select[count] = test_target_tensor[ii]
                index_select.append(test_target_index[ii])
                count += 1


        if count != 0:
            output_select = output_select[:count, :]
            target_select = target_select[:count]

            output_prob = softmax(output_select[:, :n_categories])

            for n, target in enumerate(target_select.data):
                find = ""
                target_asterisked = corpus.test_dictionary[target][0]
                target_index = index_select[n]
                
                # handle with special characters('[', ']', '(', ')') where '\' needs to be added before using re.
                for i in target_asterisked.lower():
                    if i == '●':
                        find += "\w"
                    elif i == '[':
                        find += '\['
                    elif i == ']':
                        find += '\]'
                    elif i == '(':
                        find += '\('
                    elif i == ')':
                        find += '\)'                        
                    else:
                        find += i
                pattern = re.compile(find)
                pattern_matched = False

                num = 0
                correct = False
                result = ""
                all_candidates = {}

                digit_check = 0
                for rand in target_asterisked:
                    if rand in string.digits:
                        digit_check = 1

                if digit_check == 0:
                    top_n, top_i = output_prob[n].data.topk(len(output_prob[n]))
                    category_i = top_i.cpu().numpy()

                    for i in range(0, len(output_prob[n])):
                        curr_candidate = corpus_load.dictionary.idx2word[category_i[i]]

                        if len(target_asterisked) == len(curr_candidate):
                            # pattern set to lowercase for case insensitivity
                            plausible_candidate = re.match(pattern, curr_candidate)
                            # check if candidate matches the asterisked pattern
                            if plausible_candidate:
                                curr_candidate_prob = output_prob[n].data[category_i[i]]
                                if num == 0:
                                    num += 1
                                    # test_data_list contains capitalized words
                                    if curr_candidate.lower() == test_data_list[target].lower():
                                        print ("#yes", target_asterisked, curr_candidate, test_data_list[target])
                                        correct_count += 1
                                        correct = True
                                        # if case-insensitively matched, update candidate word with the target cases
                                        curr_candidate = test_data_list[target]
                                    else:
                                        print ("#no", target_asterisked, curr_candidate, test_data_list[target])
                                    all_candidates[curr_candidate] = curr_candidate_prob
                                    temp2.append(target_asterisked)
                                elif num >= 1:
                                    num += 1
                                    if curr_candidate.lower() == test_data_list[target].lower():
                                        print (str(num), target_asterisked, curr_candidate, test_data_list[target])
                                        # update current candidate with correct cases even if it's not top candidate
                                        curr_candidate = test_data_list[target]
                                        if curr_candidate in all_candidates:
                                            all_candidates[curr_candidate] += curr_candidate_prob
                                        else:
                                            all_candidates[curr_candidate] = curr_candidate_prob
                                    else:
                                        print (str(num), target_asterisked, curr_candidate, test_data_list[target])
                                        all_candidates[curr_candidate] = curr_candidate_prob
                                pattern_matched = True

                if not pattern_matched:
                    print ("not matched", target_asterisked)
                    temp2.append(target_asterisked)
                    # if there isn't a pattern match, designate prediction as asterisked word and confidence as 0
                    all_candidates[target_asterisked] = 0
                    not_in_fixed_list_count += 1

                sorted_candidates = sorted(all_candidates.items(), key=operator.itemgetter(1), reverse=True)
                sum_conf = sum(x[1] for x in sorted_candidates)
                averages.append(all_candidates)

                total_count += 1
                progress_counter+=1
                # progress(progress_counter, total_counter, status="Word LSTM Progress:")

                if top_n.cpu().numpy()[0] > 0:
                    high_total += 1
                    if category_i[0] == target:
                        high_correct += 1

            output_log = torch.log(output_prob)

    elapsed = time.time() - start_time

    start_time = time.time()

    return correct_count / total_count, high_correct / high_total, high_total / total_count

# Loop over epochs.
lr = args.lr
best_val_loss = None

# Run on test data.
test_accuracy, high_accuracy, high_percentage = test()
print ("test_accuracy = ", test_accuracy)

# sort each word based on the order
for i in range(0, len(temp1)):
    temp3[temp1[i]] = averages[i]

for k in sorted(temp3.keys()):
    temp4[k] = temp3[k]

temp5 = []
for k in temp4:
    temp5.append(temp4[k])

dot_file = open('word_data/word_second_step_avgs', 'wb')
pickle.dump(temp5, dot_file)
dot_file.close()
print (temp5)
