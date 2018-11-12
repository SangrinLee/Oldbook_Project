# Oldbook_project

========== Main code ========== 

• predict.py : Run automatically the scripts in sequence depending on user’s option(training the language model or predicting the words). After results from unigram/LSTM are serialized, the script saves the predicted words and their probabilities as well as the modified xmls.

========== Unigram Model ========== 
freq_get_input.py : reads the XML files and extracts the words from it 
freq_train.py : uses the hash tables with words and frequencies already premade and runs the unigram test to serialize prediction probabilities for weighting later in predict.py 
freq_get_output.py : gathers unigram results from freq_train and makes a new XML that updates words that have a probability above .75 and leaves the others dotted as they were before 
freq_train_p.py : pseudo-correction for unigram 
freq_train_real.py : pseudo-correction for unigram 

========== LSTM Model ========== 
word_lstm_model.py : LSTM class 
word_preprocess.py : Converts XML to create plain text with just the words called old_books.txt under the 'word_data' directory. It creates whole corpus using word_corpus_data.py script, uses 9/10 for training set, 1/10 for validation set, all dot words for testing set. 
word_corpus_data.py : get the dictionary of all unique words, adjusts the size(50000) of it based on highest frequency, and generates the list of old book texts applying the fixed number of unique ids. By using customized thread, it finally generates training words. It also generates testing set by storing both index and dot words in the context. 
word_preprocess_pseudo.py : same as preprocessing.py but uses pseudo_correction.xml as an input from the unigram model. 
word_corpus_data_pseudo.py : same as word_corpus_data.py, but uses pseudo_correction.xml as an input, creates pseudo testing set with elements of confidence less than .75. 
word_train.py : feeds the dataset created in the preprocessing step into the model, train & evaluate the lstm model. 
word_test.py : loads the testing set created in the preprocessing step, model conditionally chosen based on the user's input, predicts the dot words, stores prediction and their corresponding probabilities. 
word_test_pseudo.py : same as word_test.py, but writes lstm pseudo correction result of each word into the xml. 
word_get_output.py : based on the result of weighting between unigram and lstm, it creates final mls, supplemental excel files. 
========== DIRECTORIES ========== 
XML : Original xmls 
freq_data : user's input, the plain text, and preprocessed text files from the original xmls. 
freq_output : the result from the 1st step of unigram model. 
freq_output_final : the result from the 2nd step of unigram model with considering dot words indicating two characters. 
word_data : the plain text from the xmls under freq_output_final directory, corpus including training & testing set, and result from the lstm. 
word_final_output : the final xml outputs. 
word_ptb_models : already trained 10 models, and their corpus information including their unique ids(dictionary)
