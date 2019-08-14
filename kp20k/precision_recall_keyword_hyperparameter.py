# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 06:10:37 2019

@author: 1000256731
"""
'''
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

lemmatizer = WordNetLemmatizer()
nltk_stopwords = nltk.corpus.stopwords.words('english');
doc2 = open('C-1.key');
    
keyword_2 = [];

for line in doc2.readlines():
    line = line.rstrip("\n\r");
    tokens = nltk.tokenize.word_tokenize(line);
    line = " ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens if not (w in nltk_stopwords)])
    keyword_2.append(line);

print(keyword_2);
'''
import spacy
import os
import csv

def calc_precision_recall(file_name, window_size, algo_name):
    # Initialize spacy 'en' model, keeping only tagger component needed for lemmatization
    nlp = spacy.load('en', disable=['parser', 'ner']);
    ##
    # generated key is the directory with generated keyword files
    ##
    file_name_1 = 'for_validation/generated_key_'+ algo_name + '_' + str(window_size) + '/' + file_name;
    #print(file_name_1);
    doc1 = open(file_name_1, encoding='utf-8');
    keyword_1 = []
    for line in doc1.readlines():
        line = line.rstrip("\n\r").lower();
        doc = nlp(line);
        line = " ".join([token.lemma_ for token in doc if not token.is_stop])
        keyword_1.append(line);
 
    
    ##
    # true key is the directory with ground truth files
    ##
    file_name_2 = 'true_dev_key/'+file_name;
    doc2 = open(file_name_2, encoding = 'utf-8');
    keyword_2 = [];
    
    for line in doc2.readlines():
        line = line.rstrip("\n\r");
        doc = nlp(line);
        line = " ".join([token.lemma_ for token in doc if not token.is_stop])
        keyword_2.append(line);
    
    if(len(keyword_2) == 0):
        return -1,-1;
    
    if(len(keyword_1)==0):
        print("HIT!");
        print(file_name);
        return -1,-1;
    
    true_positive = 0;
    false_positive = 0;
    for search_query in keyword_1:

        if(search_query in keyword_2):
            true_positive += 1;
        else:
            false_positive += 1;
    
    false_negative = len(keyword_2) - true_positive;
    precision = true_positive/(true_positive+false_positive);
    recall = true_positive/(true_positive+false_negative);
    return precision,recall;

def save_to_csv(algorithm_name, best_parameter):
    csv_file = open("params.csv",'w',newline='');
    csv_writer = csv.writer(csv_file);
    
    for i in range(len(algorithm_name)):
        row = [algorithm_name[i], best_parameter[i]];
        csv_writer.writerow(row);
    csv_file.close();
    
        

##
# true key is the directory with ground truth files
##

path = "true_dev_key/";
algorithm_name = ['singlerank','sgrank','textrank'];
window_length_array = [2,6,10, 14, 20];
best_parameter = [];
for algo_name in algorithm_name:
    max_f1 = -1;
    max_window = 0;
    for window_size in window_length_array:
        total_files = 0;
        total_precision = 0;
        total_recall = 0;
        #print(os.walk(path));
        for r,d,f in os.walk(path):
            for file in f:
                if '.key' in file:
                   precision, recall = calc_precision_recall(file, window_size, algo_name);
                   if(precision ==-1 or recall == -1):
                       continue;
                   else:
                       total_precision += precision;
                       total_recall += recall;
                       total_files += 1
        avg_precision = total_precision / total_files;
        avg_recall = total_recall/ total_files;
        f1_score = (2*avg_precision*avg_recall)/(avg_precision+avg_recall);
        #print("For the algorithm: ", algo_name," For window size: ",window_size," Avg precision: ",avg_precision," Avg recall: ",avg_recall, "F1 score: ", f1_score);
        if(f1_score>max_f1):
            max_f1 = f1_score;
            max_window = window_size;
    best_parameter.append(max_window);

save_to_csv(algorithm_name, best_parameter);
           
           

