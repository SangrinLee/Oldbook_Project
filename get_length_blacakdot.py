import pickle
import string
import numpy as np
import argparse
import threading
import word_corpus_data
import string
import glob
import re

dot_words = []
test_words = []
actual_words = []
index_words = []

test_word_count = 0

with open("freq_data/word_list.txt", "r", encoding='UTF-8') as read_file:
	word_list = read_file.read().split(" ")
	for word in word_list:
		if "●" in word:
			dot_words.append(word)

	# dot_words = dot_words
	# print (len(dot_words))
	# exit()
	word_index = 0

	for word in word_list:
		if test_word_count >= 2000:
			break
		if "●" in word:
			continue
		else:
			w_count = 0
			# print ("ORIGINAL : ", dot_words)
			for dot_word in dot_words:
				if dot_word != "None":
					dot_word_idx = dot_word.find("●")
					word_as_list = list(word)
					
					if len(dot_word) != len(word):
						w_count += 1
						continue
					else:
						word_as_list[dot_word_idx] = "●"
						test_word = ''.join(word_as_list)						

						if test_word == dot_word:
							test_word_count += 1
							print (test_word, dot_word, word, word_index, test_word_count)
							word_as_list[dot_word_idx] = "●"
							test_words.append(test_word)
							actual_words.append(word)
							index_words.append(word_index)

							dot_word = "None"
							dot_words[w_count] = "None"
							break
				w_count += 1			
		word_index += 1

print (test_words)
print (test_word_count)
print (len(test_words))

# Stores test_words
with open('freq_data/new_blackdot', 'wb') as write_file:
	pickle.dump(test_words, write_file)

# Stores actual_words
with open('freq_data/correct_blackdot', 'wb') as write_file:
	pickle.dump(actual_words, write_file)

