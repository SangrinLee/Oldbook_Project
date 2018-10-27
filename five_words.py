import xml.etree.ElementTree as ET
from xml.dom import minidom

idx_list = []
tree = ET.parse('word_data/pseudo_correction.xml')
root = tree.getroot()
item_iter = root.iter('item')

with open("freq_data/word_list.txt", "r", encoding='UTF-8', newline='') as read_file:
	words = read_file.read().split(" ")

	for word in root.iter("item"):
	    
	    element_word = word.find("word").text
	    element_index = int(word.find("index").text)

	    element_unigram_correct = word.find("unigram").find("correct").text
	    element_lstm_correct = word.find("lstm").find("correct").text

	    item = word

	    if (element_unigram_correct == "true" and element_lstm_correct == "false") or (element_unigram_correct == "false" and element_lstm_correct == "true"):
	        idx_list.append(element_index)
	        i = element_index
	        five_words_before = ET.SubElement(item, 'five_words_before')
	        five_words_before.text = words[i-5] + ", " + words[i-4] + ", " + words[i-3] + ", " + words[i-2] + ", " + words[i-1]
	        five_words_after = ET.SubElement(item, 'five_words_after')
	        five_words_after.text = words[i+1] + ", " + words[i+2] + ", " + words[i+3] + ", " + words[i+4] + ", " + words[i+5]

xmlstr = '\n'.join([line for line in minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ").split('\n') if line.strip()])
with open('word_data/pseudo_correction2', "w", newline='', encoding='UTF-8') as f:
	f.write(xmlstr)

# print ("total indices")
# print (idx_list)

# with open("freq_data/word_list.txt", "r", encoding='UTF-8', newline='') as read_file:
# 	words = read_file.read().split(" ")
# 	for i in idx_list:
# 		print (i, words[i])
# 		print ("<five_words_before>" + words[i-5] + ", " + words[i-4] + ", " + words[i-3] + ", " + words[i-2] + ", " + words[i-1] + "</five_words_before>")
# 		print ("<five_words_after>" + words[i+1] + ", " + words[i+2] + ", " + words[i+3] + ", " + words[i+4] + ", " + words[i+5] + "</five_words_after>")