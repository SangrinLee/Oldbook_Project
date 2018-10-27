import pickle
import math

# Based on the following purpose, set the model to "train", "test", or "pseudo"
# training the model : "train"
# testing the model : "test"
# pseudo correction test : "pseudo"
model = "real"

print ("========== UNIGRAM START ==========")
from freq_get_input import *
if model == "pseudo":
	from freq_train_p import *
elif model == "real":
	from freq_train_real_p import *
else:
	from freq_train import *
	from freq_get_output import *
print ("========== UNIGRAM END ==========")
# exit()
print ("========== LSTM START ==========")
if model == "pseudo":
	from word_preprocess_pseudo import *
	from word_test_pseudo_check import *
elif model == "real":
	from word_preprocess_pseudo import *
	from word_test_pseudo_real_check import *	
elif model == "train":
	from word_preprocess import *
	from word_train import *
	exit()
elif model == "test":
	from word_preprocess import *
	from word_test import *
print ("========== LSTM END ==========")
# exit()
# load the unigram averages
with open('freq_data/uni_second_step_avgs','rb') as f:
     unigram_avg=pickle.load(f)
# load the lstm averages
with open('word_data/word_second_step_avgs','rb') as b:
     lstm_avg=pickle.load(b)

print (len(unigram_avg))
print (len(lstm_avg))
# exit()

# with open('word_data/test_data2', 'rb') as f:
#     test_data_list=pickle.load(f)
with open('word_data/corpus', 'rb') as handle:
    corpus2 = pickle.load(handle)

test_data_list = corpus2.actual_word_list

for i in unigram_avg:
	if sum(i.values())!=0:
		factor=1.0/sum(i.values())
		for k in i:
			i[k]=i[k]*factor

for i in lstm_avg:
	if sum(i.values())!=0:
		factor=1.0/sum(i.values())
		for k in i:
			i[k]=i[k]*factor

# exit()
word_prob = []
new_array = []
maximum = 0
counter=-1
temp=0
word = ""
checker=0
holdword=""
stopcount=0



# print (test_data_list)
total = len(test_data_list)
print (total)
correct = 0
for i in range(len(test_data_list)):
# for i in range(len(lstm_avg)):
	# print ("** test word = ", i, test_data_list[i])
	maximum = 0
	stopcount=0
	checker = 0
	word = ""

	for key, value in unigram_avg[i].items():
		if stopcount==0:
			holdword=key
		stopcount=1

		for key1, value1 in lstm_avg[i].items():
			if key.lower() == key1.lower():
				temp = ((value) + value1) / 2
				print (key, key1, value, value1, temp, maximum)
				if temp >= maximum:
					word = key1
					maximum=temp
				checker=1

	if checker==0:
		word_prob.append(0)
		new_array.append(holdword)
	else:
		if word.lower() == test_data_list[i].lower():
			# print (word, test_data_list[i])
			correct += 1
		word_prob.append(maximum)
		new_array.append(word)

print ("total = ", total, ", correct = ", correct, ", accurac = ", correct / total)

dot_file=open('freq_data/predicted', 'wb')
pickle.dump(new_array, dot_file)
dot_file.close()
prob_file=open('freq_data/probabilities', 'wb')
pickle.dump(word_prob, prob_file)
prob_file.close()

from word_get_output import *
