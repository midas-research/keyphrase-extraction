import textacy
import textacy.keyterms
import os
import csv
import kp20k_process_predict

os.mkdir('for_predict/');

def read_params_csv():
    csv_file = open('params.csv','r');
    csv_read = csv.reader(csv_file);
    params_dict = {};
    for readlines in csv_read:
        params_dict[readlines[0]] = int(readlines[1]);
    return params_dict

def write_key_sgrank(path, file_name, num_keys, win_width):
    read_path = path + file_name + ".txt";
    doc = open(read_path,'r',encoding='utf-8');
    doc = textacy.make_spacy_doc(doc.read());
    keyword = textacy.keyterms.sgrank(doc, ngrams=(1, 2, 3, 4, 5, 6), normalize='lemma', window_width=win_width, n_keyterms=num_keys, idf=None)
    out_directory ='for_predict/generated_key_sgrank@'+ str(num_keys) +'/';
    if(os.path.isdir(out_directory) == 0):
        os.mkdir(out_directory);
    out_path = out_directory + file_name + '.key';
    out_doc = open(out_path,'w',encoding='utf-8');
    for key in keyword:
        out_doc.write(key[0].lower());
        out_doc.write("\n");

def write_key_singlerank(path, file_name, num_keys, win_width):
    read_path = path + file_name + ".txt";
    doc = open(read_path,'r',encoding='utf-8');
    doc = textacy.make_spacy_doc(doc.read(), lang='en');
    keyword = textacy.keyterms.key_terms_from_semantic_network(doc, normalize='lemma',window_width=win_width, edge_weighting="cooc_freq", ranking_algo="pagerank", join_key_words=True,n_keyterms=num_keys)
    out_directory ='for_predict/generated_key_singlerank@'+ str(num_keys) +'/';
    if(os.path.isdir(out_directory) == 0):
        os.mkdir(out_directory);
    out_path = out_directory + file_name + '.key';
    out_doc = open(out_path,'w',encoding='utf-8');
    for key in keyword:
        out_doc.write(key[0].lower());
        out_doc.write("\n");

def write_key_textrank(path, file_name, num_keys, win_width):
    read_path = path + file_name + ".txt";
    doc = open(read_path,'r',encoding='utf-8');
    doc = textacy.make_spacy_doc(doc.read());
    keyword = textacy.keyterms.key_terms_from_semantic_network(doc, normalize='lemma',window_width=win_width, edge_weighting="binary", ranking_algo="pagerank", join_key_words=True,n_keyterms=num_keys)
    out_directory ='for_predict/generated_key_textrank@'+ str(num_keys) +'/';
    if(os.path.isdir(out_directory) == 0):
        os.mkdir(out_directory);
    out_path = out_directory + file_name + '.key';
    out_doc = open(out_path,'w',encoding='utf-8');
    for key in keyword:
        out_doc.write(key[0].lower());
        out_doc.write("\n");

path = 'test/';
params_dict = read_params_csv();
n_keyterms = [5,10,15];

for num_keys in n_keyterms:
    for r,d,f in os.walk(path):
        for file in f:
            if '.txt' in file:
                print("PROCESSING FOR FILE: ",file);
                write_key_sgrank(path,file.rstrip(".txt"),num_keys,params_dict['sgrank']);
                write_key_singlerank(path,file.rstrip(".txt"),num_keys,params_dict['singlerank']);
                write_key_textrank(path,file.rstrip(".txt"),num_keys,params_dict['textrank']);
                

import precision_recall_keyword
import shutil

shutil.rmtree('for_predict/');