import string
from random import randint
import re
import numpy
import operator
import pickle
import sys
import time

def progress(count, total, status = ""):
    bar_len = 30
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  

def generate_test_words(input_filename, output_filename):
	with open(input_filename, 'r', encoding="UTF-8") as read_file:
		word_list = read_file.read().split()

	with open(output_filename, 'w', encoding="UTF-8") as write_file:
		for word in word_list:
			if "●" in word:
				write_file.write(str(word) + " ")

generate_test_words("freq_data/word_list.txt", "freq_data/test_words.txt")

with open("freq_data/test_words.txt", "r", encoding="UTF-8") as read_file:
	test_words = read_file.read().split()

# Ask users to enter the year
year = input('Enter the year : ')
if int(year) <= 1640:
	with open('freq_data/hash_table_1600','rb') as f:
		word_hash_table = pickle.load(f)
else:
	with open('freq_data/hash_table_1640','rb') as f:
		word_hash_table = pickle.load(f)

# Stores the year
with open('freq_data/user_input', 'wb') as write_file:
	pickle.dump([year], write_file)


def test_word_model(test_words, delta, word_hash_table):
	progress_counter = 0

	probabilities = []
	predicted_word_list = []

	average_dots = []
	total_average_dots = []
	One_hundred_percent = 0
	confidentcounter1 = 0
	confidentcounter2 = 0
	confidentcounter3 = 0
	confidentcounter4 = 0
	confidentcounter5 = 0
	confidentcounter6 = 0
	confidentcounter7 = 0
	confidentcounter8 = 0
	confidentcounter85 = 0
	confidentcounter75 = 0
	confidentcounter65 = 0
	confidentcounter55 = 0

	deltasums=[]
	freq_average95 = 0
	freq_average9 = 0
	freq_average85 = 0
	freq_average8 = 0
	freq_average75 = 0
	freq_average7 = 0
	freq_average65 = 0
	freq_average6 = 0
	freq_average55 = 0
	freq_average5 = 0
	freq_average4 = 0
	freq_average3 = 0

	for test_word in test_words:
		progress(progress_counter, len(test_words), status="Unigram Progress:")
		progress_counter += 1

		predictions = {}
		has_predictions = False

		changecheck = 0
		score = 0
		wordlength = len(test_word)
	
		has_num = False
		for digit in string.digits:
			if digit in test_word:
				has_num = True
				break

		if has_num == False:
			for hash_word in word_hash_table:
				has_hash_word = False
				if len(test_word) == len(hash_word):
					if len(test_word) == 1:
						has_hash_word = True
					else:
						for x in range(len(test_word)):
							if test_word[x] == hash_word[x]:
								has_hash_word = True
								break

				if has_hash_word:
					dot_index_list = [m.start() for m in re.finditer('●', test_word)]
					test_word_plain = test_word.replace("●", "")
					hash_word_plain = hash_word
					counter = 0
					for x in dot_index_list:
						hash_word_plain = hash_word_plain[:x-counter] + hash_word_plain[(x+1-counter):]
						counter += 1
					if test_word_plain == hash_word_plain:
						predictions[test_word + " " + hash_word + " "] = word_hash_table[hash_word]
						has_predictions = True

		if has_predictions:
			maximum = max(predictions, key=predictions.get)
			probability = (predictions[maximum] - delta[wordlength]) / sum(predictions.values())
			probabilities.append(probability)

			if probability > .3:
				confidentcounter1 += 1
				freq_average3 += probability
			if probability > .4:
				confidentcounter2 += 1
				freq_average4 += probability
			if probability > .5:
				confidentcounter3 += 1
				freq_average5 += probability
			if probability > .6:
				confidentcounter4 += 1
				freq_average6 += probability
			if probability > .7:
				confidentcounter5 += 1
				freq_average7 += probability
			if probability > .8:
				confidentcounter6 += 1
				freq_average8 += probability
			if probability > .9:
				confidentcounter7 += 1
				freq_average9 += probability
			if probability > .95:
				confidentcounter8 += 1
				freq_average95 += probability
			if probability == 1:
				One_hundred_percent += 1
			if probability > .85:
				confidentcounter85 += 1
				freq_average85 += probability
			if probability > .75:
				confidentcounter75 += 1
				predicted_word_list.append(maximum.split(" ")[1])
				changecheck = 1
				freq_average75 += probability
			if probability > .65:
				confidentcounter65 += 1
				freq_average65 += probability
			if probability > .55:
				confidentcounter55 += 1
				freq_average55 += probability
			# print("Max Frequency:", predictions[maximum]," Percentage frequency over matches:",str(round(100*(predictions[maximum]-delta[wordlength])/sum(predictions.values()),2))+"%")

		if not has_predictions:
			probabilities.append(0)
			first_dot = False
			first_check = 0
			two_dots = False
			for rand in test_word:
				if first_check == 1 and rand == "●":
					first_check = 0
					break
				if rand == "●" and test_word[0] == "●":
					first_dot = True
					first_check = 1

			if first_check == 1 and first_dot == True:
				test_word = test_word.replace("●","●●",1)

		if not has_predictions:
			dictt={}
			dictt[test_word]=0
			average_dots.append(dictt)
			total_average_dots.append(dictt)
		else:
			hold_dict={}
			max_size=0
			saved=""
			prediction_size = sum(predictions.values())
			total_sum = sum(predictions.values())
			while(prediction_size!=0 and max_size<5):
				saved=maximum.split(" ")[1].split(" "[0])[0]
				hold_dict[saved]=predictions[maximum]/total_sum
				predictions_size = prediction_size-1
				max_size+=1
				predictions[maximum]=.5
				maximum=max(predictions, key=predictions.get)
			total_average_dots.append(hold_dict)

		if changecheck==0:
			predicted_word_list.append(test_word)
			hold_dict={}
			max_size=0
			saved=""
			if has_predictions:
				prediction_size=sum(predictions.values())
				total_sum=sum(predictions.values())
				while(prediction_size!=0 and max_size<5):
					saved=maximum.split(" ")[1].split(" "[0])[0]
					hold_dict[saved]=predictions[maximum]/total_sum
					predictions_size = prediction_size-1
					max_size+=1
					predictions[maximum]=.5
					maximum=max(predictions, key=predictions.get)
				average_dots.append(hold_dict)

	with open('freq_data/predicted', 'wb') as write_file:
		pickle.dump(predicted_word_list, write_file)
	
	second_dot_file=open('freq_data/uni_second_step_avgs', 'wb')
	pickle.dump(average_dots, second_dot_file)
	second_dot_file.close()

	prob=open('freq_data/probabilities', 'wb')
	pickle.dump(probabilities, prob)
	prob.close()

	supp_probs=open('freq_data/sup_probs', 'wb')
	pickle.dump(total_average_dots, supp_probs)
	supp_probs.close()

deltas = [0.2, 0.33, 0.26, 0.27, 0.21, 0.09, 0.09, 0.04, 0.04, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
test_word_model(test_words, deltas, word_hash_table)
