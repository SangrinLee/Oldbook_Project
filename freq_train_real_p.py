import string
from random import randint
import re
import numpy
import operator
import pickle
import sys
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom

XML_output = "freq_data/pseudo_correction.xml"
actual_word_list = []

def progress(count, total, status = ""):
    bar_len = 30
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  

# randomly assign one asterisks to 200 random non-black-dot word in input file
def generate_pseudo_correction_task(input, output, XML_output):
    # build an XML tree
    doc = ET.Element('document')
    input_file = open(input, 'r', newline='', encoding='UTF-8')
    file_contents = input_file.read()
    input_file.close()
    words = file_contents.split()
    file = open(output, 'w', newline='', encoding='UTF-8')
    i = 0

    # added
    total_words = len(words)
    index = 0

    while index < total_words:
        # index = randint(0, len(words) - 1) # deleted
        # if (not re.search('●', words[index])) and (not re.search('[0-9]+', words[index])) and len(words[index]) > 1:
        if re.search('●', words[index]):
            item = ET.SubElement(doc, 'item')
            item_word = ET.SubElement(item, 'word')
            item_word.text = words[index]
            item_index = ET.SubElement(item, 'index')
            item_index.text = str(index)
            word = words[index]
           	
           	# deleted
            # rand_num = randint(0, len(word) - 1)
            # word = word[:rand_num] + "●" + word[rand_num + 1:]
            
            # added
            # pick_num = (len(word) - 1) // 2
            # word = word[:pick_num] + "●" + word[pick_num + 1:]
            # print (word)
            
            item_asterisked = ET.SubElement(item, 'asterisked')
            item_asterisked.text = word
            print("i :", i, ", index :", index, ", test_word :", word, ", actual_word :", words[index])
            actual_word_list.append(words[index])
            file.write(str(word) + " ")
            i += 1
        index += 1 # added

    file.close()
    xmlstr = minidom.parseString(ET.tostring(doc)).toprettyxml(indent="    ")
    with open(XML_output, "w", newline='', encoding='UTF-8') as f:
        f.write(xmlstr)

    #tree = ET.ElementTree(doc)
    #tree.write(XML_output)


def generate_test_words(input_filename, output_filename):
	with open(input_filename, 'r', encoding="UTF-8") as read_file:
		word_list = read_file.read().split()

	with open(output_filename, 'w', encoding="UTF-8") as write_file:
		for word in word_list:
			if "●" in word:
				write_file.write(str(word) + " ")

generate_pseudo_correction_task("freq_data/word_list.txt", "freq_data/test_words.txt", XML_output)

# generate_test_words("freq_data/word_list.txt", "freq_data/test_words.txt")

with open("freq_data/test_words.txt", "r", encoding="UTF-8") as read_file:
	test_words = read_file.read().split()

# Ask users to enter the year
# year = input('Enter the year : ')
with open('freq_data/date', 'rb') as handle:
    year = pickle.load(handle)
    
if int(year) <= 1640:
	with open('freq_data/hash_table_1600','rb') as f:
		word_hash_table = pickle.load(f)
else:
	with open('freq_data/hash_table_1640','rb') as f:
		word_hash_table = pickle.load(f)


# Stores the year
with open('freq_data/user_input', 'wb') as write_file:
	pickle.dump([year], write_file)

total_counter=len(test_words)

def test_word_model(test_words, delta, word_hash_table):
	idx = 0
	confidence_correct = 0
	confidence_total_counter=0
	correct=0

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

	tree = ET.parse(XML_output)
	doc = tree.getroot()
	item_iter = doc.iter('item')

	for test_word in test_words:
		# defaults to empty <item> element to prevent StopIteration exception
		curr_item = next(item_iter, ET.Element('item'))
		item_unigram = ET.SubElement(curr_item, 'unigram')

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
			# sort predictions by normalized (delta?) confidence
			sorted_predictions = sorted(predictions.items(), reverse=True, key=operator.itemgetter(1))

			# append to XML <unigram> element
			item_prediction = ET.SubElement(item_unigram, 'prediction')
			item_prediction.text = maximum.split(" ")[1]
			item_correct = ET.SubElement(item_unigram, 'correct')
			item_confidence = ET.SubElement(item_unigram, 'confidence')
			item_confidence.text = str((predictions[maximum]-delta[wordlength])/sum(predictions.values())) # added
			item_candidates = ET.SubElement(item_unigram, 'candidates')
			n = 0
			while n < 5 and n < len(sorted_predictions):
				item_candidate = ET.SubElement(item_candidates, 'candidate')
				item_name = ET.SubElement(item_candidate, 'name')
				item_name.text = sorted_predictions[n][0].split(" ")[1]
				item_conf = ET.SubElement(item_candidate, 'conf')
				# print ("$$$$$", sorted_predictions[n][1], sum(predictions.values()))
				# print (str((sorted_predictions[n][1]-delta[wordlength])/sum(predictions.values())))
				item_conf.text = str((sorted_predictions[n][1]-delta[wordlength])/sum(predictions.values())) # added

				n += 1

			probability = (predictions[maximum] - delta[wordlength]) / sum(predictions.values())

			print ("#####", predictions[maximum], delta[wordlength], sum(predictions.values()), probability)

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
			if probability >= 0:
				confidentcounter8 += 1
				freq_average95 += probability

				predicted_word_list.append(maximum.split(" ")[1])
				changecheck = 1
				if maximum.split(" ")[1] == actual_word_list[idx]:
					confidence_correct += 1
				confidence_total_counter += 1

				item_conf_correct = ET.SubElement(item_unigram, 'conf_correct')
				item_conf_correct.text = maximum.split(" ")[1]

			if probability == 1:
				One_hundred_percent += 1
			if probability > .85:
				confidentcounter85 += 1
				freq_average85 += probability
			if probability > .75:
				confidentcounter75 += 1
				# predicted_word_list.append(maximum.split(" ")[1])
				# changecheck = 1
				# freq_average75 += probability

				# if maximum.split(" ")[1] == actual_word_list[idx]:
				# 	confidence_correct += 1
				# confidence_total_counter += 1

			if probability > .65:
				confidentcounter65 += 1
				freq_average65 += probability
			if probability > .55:
				confidentcounter55 += 1
				freq_average55 += probability
			# print("Max Frequency:", predictions[maximum]," Percentage frequency over matches:",str(round(100*(predictions[maximum]-delta[wordlength])/sum(predictions.values()),2))+"%")

		if not has_predictions:
			# append to XML <unigram> element
			item_prediction = ET.SubElement(item_unigram, 'prediction')
			item_prediction.text = "None"
			item_correct = ET.SubElement(item_unigram, 'correct')
			item_correct.text = "AA"
			item_confidence = ET.SubElement(item_unigram, 'confidence')
			item_confidence.text = "0"
			item_candidates = ET.SubElement(item_unigram, 'candidates')
			item_candidates.text = "None"
			print ("NO PREDICTION", actual_word_list[idx])
			idx += 1
			dictt={}
			dictt[test_word]=0
			total_average_dots.append(dictt)
			continue
		# print (has_predictions, actual_word_list[idx])
		if maximum.split(" ")[1] == actual_word_list[idx]:
			print("#yes", "Test Word:", maximum.split(" ")[0], ", Prediction:", maximum.split(" ")[1], ", Actual Word:", 
				actual_word_list[idx])
			correct += 1
			item_correct.text = "true"
		else:
			print("no", "Test Word:", maximum.split(" ")[0], ", Prediction:", maximum.split(" ")[1], ", Actual Word:",
				actual_word_list[idx])
			item_correct.text = "false"
		idx += 1


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
				while(prediction_size!=0 and max_size<50):
					saved=maximum.split(" ")[1].split(" "[0])[0]
					hold_dict[saved]=predictions[maximum]/total_sum
					predictions_size = prediction_size-1
					max_size+=1
					predictions[maximum]=.5
					maximum=max(predictions, key=predictions.get)
				average_dots.append(hold_dict)

	print ("total_count = ", total_counter, ", correct = ", correct)
	print ("accuracy = ", correct / total_counter)

	print("confidence_total_count = ", confidence_total_counter, ", confidence_correct = ", confidence_correct)
	print("accuracy = ", confidence_correct / confidence_total_counter)


	with open('freq_data/predicted', 'wb') as write_file:
		pickle.dump(predicted_word_list, write_file)
	
	second_dot_file=open('freq_data/uni_second_step_avgs', 'wb')
	pickle.dump(total_average_dots, second_dot_file)
	second_dot_file.close()

	prob=open('freq_data/probabilities', 'wb')
	pickle.dump(probabilities, prob)
	prob.close()

	xmlstr = '\n'.join([line for line in minidom.parseString(ET.tostring(doc)).toprettyxml(indent="    ").split('\n') if line.strip()])
	with open(XML_output, "w", newline='', encoding='UTF-8') as f:
		f.write(xmlstr)

deltas = [0, 0.2, 0.33, 0.26, 0.27, 0.21, 0.09, 0.09, 0.04, 0.04, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # added
test_word_model(test_words, deltas, word_hash_table)
