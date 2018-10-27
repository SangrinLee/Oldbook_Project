# -*- coding: utf-8 -*-

import re
import glob
import string
import pickle
import csv 
import sys

with open('freq_data/predicted', 'rb') as handle:
	dot_list = pickle.load(handle)

with open('freq_data/probabilities', 'rb') as handle:
	probabilities = pickle.load(handle)

with open('freq_data/sup_probs', 'rb') as handle:
	supp_probs = pickle.load(handle)

specials = "▪〈◊〉̄íàé…§❧áùèìòúꝑÉ·ÿöóëôî¶‑—ηçćü⸫âûꝰêΩ⁎✚˙ΙΗΣΟΥ⸪“”☞ȧ†❀½¾⅓¼⅔⅛⅚⅝⅖⅗⅕ʒï☉īĕăōēĭŏāūŭꝓΨ‡ę⅘ȝπβÔÁΕΝΤΚἘΠΜΛΔÎΑ‖ě⸬κατῇλθενἐισἁδουΦχρꝙ♈♉♋♍♏♑♓♒♐♎♌♊♄♃♂♀☿☽☌⚹□△☍☊℥ωŕ☐ª╌ÇĈיוהבארäÓß☜Γº⁂חȣφγζμξψ♮⅙℈ŷέςś∵ΈΡΊ℞Βṗ☟☝ƿΘↂↁↈ£ńϹ×ꝭåÖ𝄢𝄡𝄞톼텮톺텥톹𝆹´άόῷÒÀþðꝧÈ☧Χ̇משכלדȜİ°℟℣∶⅞−ůËñῳėċżÙÛ∣☋⁙′ὈήῶῥὴꝯΞ″ꝗ⋆גזטךםנןסעפףצץקת¦ὸὶ̔ͅ⊙ṙǵ◆ÐṅÆἙ‘ΖŚÞźὅ𝄁ῦίἩὁἀ♡∷‴ὑ○♁🜕∝⅜ΌῚᾺΆ√‐✴Ú⌊∞¿Ύ▵◬🜹ἹὰÜÑŵ∴˜❍¯˘🜂✝🜔𝆶𝆷𝆸텯𝇋𝇍𝇈♯⋮☾🜄ὙἈ̂★ĉňƳƴ¡АаБбВвГгДдЕеЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЮŝǽ⁏ЫЬἜ’Â🜍ↃÌ÷∽𝇊õ♥♭⊕⊗☼ȯύÏὨẏṫ±∓∼ἰđČýḃḅ؛🜖ĥø․⁁∠ὼǔḍ¨▴–▿ʹϊ̈ↇϛϟϡ͵🝆" # ,.;\'\\1234567890&*-/"
all_words=[]

for filename in glob.glob('XML/*.xml'):
	with open(filename, "r", encoding='UTF-8', newline='') as read:
		for i in read: 
			if re.match(".*lemma=.*", i):
				word = i.split("</w>")[0].split(">")[1]
				all_words.append(word)

count = 0
count1 =0
tot = 0
xml_id=[]
xml_id.append("XML_ID")
last_five=[]
last_five.append("Previous_Five_Words")
next_five=[]
next_five.append("Next_Five_Words")
probs=[]
dot_word=[]
dot_word.append("Dot_Word")
correct_word=[]
correct_word.append("Predicted_Word")
correct_prob=[]
correct_prob.append("First_Probability")
second_prediction=[]
second_prediction.append("Second_Predicted_Word")
second_prob=[]
second_prob.append("Second_Probability")
third_prediction=[]
third_prediction.append("Third_Predicted_Word")
third_prob=[]
third_prob.append("Third_Probability")
first_word=""
second_word=""
third_word=""
fourth_word=""
fifth_word=""
sixth_word=""
seventh_word=""
eighth_word=""
ninth_word=""
tenth_word=""

supp_counter=-1
word_count=-1
slash = input('\n Enter 1 if Mac/Others, 2 if Windows, if index error, use the other option:')
# slash = 2
slash1=""
if int(slash) ==1:
	slash1="/"
else:
	slash1="\\"

winmac = []
winmac.append(slash1)
user_input=open('freq_data/winmac', 'wb')
pickle.dump(winmac, user_input)
user_input.close()

for filename in glob.glob('XML/*.xml'):
	with open("freq_output/" + filename.split(slash1)[1].split(".")[0] + ".xml", "w", encoding='UTF-8', newline='') as write_file:
		with open(filename, "r", encoding='UTF-8', newline='') as read:
			for i in read:
				if re.match(".*lemma=.*", i):
					word = i.split("</w>")[0].split(">")[1]
					if "●" in word:
						supp_counter+=1
					word_count+=1
					first_word=second_word
					second_word=third_word
					third_word=fourth_word
					fourth_word=fifth_word
					fifth_word=sixth_word
					sixth_word = word
					original = word
					text_list = (re.compile(r'●').findall(word))

					if len(text_list) > 0 and probabilities[count1]>=0.75:
						dot_word.append(word)
						word = word.replace(word, dot_list[count], 1)
						j=""
						j=i
						j=j.split("xml:id=\"")[1].split("\"")[0]

						first5=[]
						second5=[]
						first5.append([first_word,second_word,third_word,fourth_word,fifth_word])
						if (word_count+5<=len(all_words)):
							second5.append([all_words[word_count+1],all_words[word_count+2],all_words[word_count+3],all_words[word_count+4],all_words[word_count+5]])

						correct_word.append(word)
						xml_id.append(j)
						it_counter=-1
						change1=0
						change2=0
						for key, value in supp_probs[supp_counter].items():
							it_counter+=1
							if it_counter==1:
								second_prediction.append(key)
								second_prob.append(value)
								change1=1
							if it_counter==2:
								third_prediction.append(key)
								third_prob.append(value)
								change2=1
						if change1==0:
							second_prediction.append("N/A")
							second_prob.append("0")
						if change2==0:
							third_prediction.append("N/A")
							third_prob.append("0")
						correct_prob.append(probabilities[count])
						nexttt=""
						prev = first5[0][0] + " " + first5[0][1] + " " + first5[0][2] + " " + first5[0][3] + " " + first5[0][4]
						if len(second5)!=0:
							nexttt = second5[0][0] + " " + second5[0][1] + " " + second5[0][2] + " " + second5[0][3] + " " + second5[0][4]
						else:
							nexttt="N/A"
						last_five.append(prev)
						next_five.append(nexttt)
						i = i.replace('>' + original + '</w>', ' type=\"machine-fixed\" corresp=\"' + original + '\" cert="'+
						 str(round(probabilities[count1]*100))+"%\">" + word + '</w>')
						count += 1
						count1 +=1
						write_file.write(i)
					elif len(text_list)>0 and probabilities[count1]<=0.75:
						count += 1
						count1 += 1
						write_file.write(i)
					else:
						write_file.write(i)
				else:
					write_file.write(i)

for i in range(len(correct_word)):
	dot_word[i]=dot_word[i].replace("●","*")
	last_five[i]=last_five[i].replace("●","*")
	next_five[i]=next_five[i].replace("●","*")
	next_five[i]=next_five[i].replace("〈…〉","<...>")
	last_five[i]=last_five[i].replace("〈…〉","<...>")
	
	for j in last_five[i]:
		if j in specials:
			last_five[i] = last_five[i].replace(j, "<special>")

	for j in next_five[i]:
		if j in specials:
			next_five[i] = next_five[i].replace(j, "<special>")

with open("freq_output/supplemental.csv","w", encoding='UTF-8') as output:
	writer=csv.DictWriter(output,fieldnames=["XML_ID", "Dot_Word","Previous_Five_Words", "Replacement","Next_Five_Words","First_Choice_Probability","Second_Choice_Replacement","Second_Choice_Probability","Third_Choice_Replacement", "Third_Choice_Probability"])
	for i in range (len(xml_id)):
		writer.writerow({'XML_ID':xml_id[i],'Dot_Word':dot_word[i],'Previous_Five_Words':last_five[i],'Replacement':correct_word[i],'Next_Five_Words':next_five[i],'First_Choice_Probability':correct_prob[i],'Second_Choice_Replacement':second_prediction[i],'Second_Choice_Probability':second_prob[i],'Third_Choice_Replacement':third_prediction[i],'Third_Choice_Probability':third_prob[i]})
output.close()
