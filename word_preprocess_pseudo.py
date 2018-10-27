# -*- coding: utf-8 -*-
from random import randint
import pickle
import string
import numpy as np
import argparse
import threading
import word_corpus_data_pseudo as data
from random import randint
import string
import math
import glob
import re

#################################################################
# Extract words from the xmls
#################################################################
all_letters = string.ascii_letters + string.digits + "'" + " " + "-" + '●'

with open("word_data/old_books_temp.txt", "w", encoding='UTF-8', newline='') as write_file:
	for filename in glob.glob('XML/*.xml'):
		with open(filename, "r", encoding='UTF-8', newline='') as read_file:
			first = True
			for line in read_file:
				if re.match(".*lemma=.*", line):
					word = line.split("</w>")[0].split(">")[1]
					if first == True:
						write_file.write(word)
						first = False
					else:
						write_file.write(" " + word)
				elif re.match(".*</pc>", line):
					word = line.split("</pc>")[0].split(">")[1]
					write_file.write(word)

with open("word_data/old_books.txt", "w", encoding="UTF-8") as write_file:
	with open("word_data/old_books_temp.txt", "r", encoding="UTF-8") as read_file:
		text = ""
		for ch in read_file.read():
			if ch in all_letters:
				text += ch
		
		text = " ".join(text.split())
		write_file.write(text)

#################################################################
# Clean and asterisk PTB corpus, generate test data
#################################################################

corpus = data.Corpus('word_data')
with open('word_data/corpus', 'wb') as handle:
    pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

all_data = corpus.rare_word
all_data = all_data.astype(np.int64)
with open('word_data/all_data_array', 'wb') as handle:
    pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


