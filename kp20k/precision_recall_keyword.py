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
def calc_precision_recall(file_name, algo_name, num_keys):
    # Initialize spacy 'en' model, keeping only tagger component needed for lemmatization
    nlp = spacy.load('en', disable=['parser', 'ner']);
    ##
    # generated key is the directory with generated keyword files
    ##
    file_name_1 = 'for_predict/generated_key_'+algo_name+'@'+str(num_keys)+'/'+file_name;
    doc1 = open(file_name_1);
    keyword_1 = []
    for line in doc1.readlines():
        line = line.rstrip("\n\r").lower();
        doc = nlp(line);
        line = " ".join([token.lemma_ for token in doc if not token.is_stop])
        keyword_1.append(line);
    if(len(keyword_1) == 0):
        print(file_name_1);
        return -1,-1;
    
    ##
    # true key is the directory with ground truth files
    ##
    file_name_2 = 'true_key/'+file_name;
    doc2 = open(file_name_2);
    keyword_2 = [];
    
    for line in doc2.readlines():
        line = line.rstrip("\n\r");
        doc = nlp(line);
        line = " ".join([token.lemma_ for token in doc if not token.is_stop])
        keyword_2.append(line);
    
    if(len(keyword_2) == 0):
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

##
# true key is the directory with ground truth files
##

path = "true_key/";
print(os.walk(path));
algo_name = ['sgrank','singlerank','textrank'];
keyword_len_list = [5,10,15];

csv_file = open('result_csv.csv','w', newline='');
csv_write = csv.writer(csv_file);

for algo in algo_name:
    for num_keys in keyword_len_list:
        total_files = 0;
        total_precision = 0;
        total_recall = 0;
        for r,d,f in os.walk(path):
            for file in f:
                if '.key' in file:
                   precision, recall = calc_precision_recall(file, algo, num_keys);
                   if(precision!=-1 or recall != -1):
                       total_precision += precision;
                       total_recall += recall;
                       total_files += 1
            avg_precision = total_precision / total_files;
            avg_recall = total_recall/ total_files;
            print(algo,"@",num_keys,": ","Average precision: ",avg_precision," average recall: ",avg_recall);
            algo_str = str(algo) + "@" + str(num_keys);
            csv_write.writerow([algo_str,avg_precision,avg_recall]);

csv_file.close();

