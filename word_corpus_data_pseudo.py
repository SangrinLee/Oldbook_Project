import os
import torch
import numpy as np
import re
import threading
import pickle
import math
from random import randint
import string
import xml.etree.ElementTree as ET
from xml.dom import minidom

class MyThread (threading.Thread):
    def __init__(self, id, data, corpus):
        threading.Thread.__init__(self)
        self.id = id
        self.data = data
        self.corpus = corpus

    def run(self):
        data_array = np.array([])

        for i in range(len(self.data) // 8 * self.id, min(len(self.data) // 8 * (self.id + 1), len(self.data))):
            pattern = re.compile(r'●')

            if pattern.findall(self.data[i]):
                data_array = np.append(data_array, self.corpus.load_corpus.dictionary.word2idx["<bd>"]) # added
               
            else:
                if self.data[i] in self.corpus.load_corpus.dictionary.idx2word:
                    data_array = np.append(data_array, self.corpus.load_corpus.dictionary.word2idx[self.data[i]]) # added
                else:
                    data_array = np.append(data_array, self.corpus.load_corpus.dictionary.word2idx["<unk>"]) # added


            if i % (len(self.data) // 50) == 0:
                print("Thread {} at {:2.1f}%".format(self.id, 100 * (i - len(self.data) // 8 * self.id) /
                      (min(len(self.data) // 8 * (self.id + 1), len(self.data)) - len(self.data) // 8 * self.id)))

        with open('word_data/rare_data_array_{}'.format(self.id), 'wb') as handle:
            pickle.dump(data_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        
        self.actual_word_list = []
        self.pseudo_word_list = []
        self.pseudo_index_list = []

        self.generate_pseudo_words()

        with open('freq_data/user_input', 'rb') as f:
            year=pickle.load(f)
        year1 = year[0]
        year_round=math.floor(int(year1)/10)
        # year_round = 800 # added
        with open('word_ptb_models/dict_array_'+str(year_round)+'1','rb') as handle:
            load_corpus = pickle.load(handle)

        self.load_corpus = load_corpus
        self.rare_word = self.generate_data_set(os.path.join(path, 'old_books.txt'))
    
    # pseudocorrection
    def generate_pseudo_words(self):
        actual_word_list = []
        i = 0
        # read from the xml
        tree = ET.parse('freq_data/pseudo_correction.xml')
        root = tree.getroot()

        for word in root.iter("item"):
            element_word = word.find("word").text
            element_index = int(word.find("index").text)
            element_asterisked = word.find("asterisked").text
            print (i, element_word, element_index, element_asterisked)
            
            element_unigram = word.find("unigram")
            element_confidence = float(element_unigram.find("confidence").text)
            if element_confidence >= 0:
                self.actual_word_list.append(element_word)
                self.pseudo_word_list.append(element_asterisked)
                self.pseudo_index_list.append([element_index, str(i)])
                i += 1

    def generate_data_set(self, path):
        assert os.path.exists(path)

        with open(path, 'r', encoding='UTF-8', newline='') as f:
            ids = np.array([])

            for line in f:
                words = line.split()
                                
                thread0 = MyThread(0, words, self)
                thread1 = MyThread(1, words, self)
                thread2 = MyThread(2, words, self)
                thread3 = MyThread(3, words, self)
                thread4 = MyThread(4, words, self)
                thread5 = MyThread(5, words, self)
                thread6 = MyThread(6, words, self)
                thread7 = MyThread(7, words, self)
                thread8 = MyThread(8, words, self)
                thread0.start()
                thread1.start()
                thread2.start()
                thread3.start()
                thread4.start()
                thread5.start()
                thread6.start()
                thread7.start()
                thread8.start()
                thread0.join()
                thread1.join()
                thread2.join()
                thread3.join()
                thread4.join()
                thread5.join()
                thread6.join()
                thread7.join()
                thread8.join()

        with open('word_data/rare_data_array_0', 'rb') as handle:
            data_array_0 = pickle.load(handle)
        with open('word_data/rare_data_array_1', 'rb') as handle:
            data_array_1 = pickle.load(handle)
        with open('word_data/rare_data_array_2', 'rb') as handle:
            data_array_2 = pickle.load(handle)
        with open('word_data/rare_data_array_3', 'rb') as handle:
            data_array_3 = pickle.load(handle)
        with open('word_data/rare_data_array_4', 'rb') as handle:
            data_array_4 = pickle.load(handle)
        with open('word_data/rare_data_array_5', 'rb') as handle:
            data_array_5 = pickle.load(handle)
        with open('word_data/rare_data_array_6', 'rb') as handle:
            data_array_6 = pickle.load(handle)
        with open('word_data/rare_data_array_7', 'rb') as handle:
            data_array_7 = pickle.load(handle)
        with open('word_data/rare_data_array_8', 'rb') as handle:
            data_array_8 = pickle.load(handle)

        data_array = np.append(data_array_0, [data_array_1, data_array_2, data_array_3,
                                          data_array_4, data_array_5, data_array_6, data_array_7])
        data_array = np.append(data_array, data_array_8)

        return data_array
