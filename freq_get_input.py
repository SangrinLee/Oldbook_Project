import string
import glob
import re
import pickle

all_letters = string.ascii_letters + string.digits + "'" + " " + "-" + '●'

with open("freq_data/old_books.txt", "w", encoding="UTF-8") as write_file:
	for filename in glob.glob('XML/*.xml'):
		with open(filename, "r", encoding="UTF-8") as read_file:
			first = True
			date_checked = False
			for line in read_file:

				if re.match(".*<date>.*</date>.*", line):
					date = line.split("<date>")[1].split("</date>")[0]
					if not date_checked:
						with open('freq_data/date', 'wb') as handle:
						    pickle.dump(date, handle, protocol=pickle.HIGHEST_PROTOCOL)
						date_checked = True

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

# word
# with open("word_data/old_books.txt", "w", encoding="UTF-8") as write_file:
# 	for filename in glob.glob('XML/*.xml'):
# 		with open(filename, "r", encoding="UTF-8") as read_file:
# 			first = True
# 			for line in read_file:
# 				if re.match(".*lemma=.*", line):
# 					word = line.split("</w>")[0].split(">")[1]
# 					if first == True:
# 						write_file.write(word)
# 						first = False
# 					else:
# 						write_file.write(" " + word)
# 				elif re.match(".*</pc>", line):
# 					word = line.split("</pc>")[0].split(">")[1]
# 					write_file.write(word)

with open("freq_data/word_list.txt", "w", encoding="UTF-8") as write_file:
	with open("freq_data/old_books.txt", "r", encoding="UTF-8") as read_file:
		text = ""
		for ch in read_file.read():
			if ch in all_letters:
				text += ch
		
		text = " ".join(text.split())
		write_file.write(text)
		