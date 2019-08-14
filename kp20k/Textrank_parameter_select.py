import spacy
import textacy
import textacy.keyterms
import os

def write_key_textrank(path, file_name, window_size):
    read_path = path + file_name + ".txt";
    doc = open(read_path,'r',encoding='utf-8');
    doc = textacy.make_spacy_doc(doc.read(),lang='en');
    keyword = textacy.keyterms.key_terms_from_semantic_network(doc, normalize='lemma',window_width=window_size, edge_weighting="binary", ranking_algo="pagerank", join_key_words=True,n_keyterms=15)
    out_directory = 'for_validation/generated_key_textrank_' + str(window_size) + '/';
    if(os.path.isdir(out_directory) == 0):
        os.mkdir(out_directory);
    out_path = out_directory + file_name + '.key';
    print(out_path);
    out_doc = open(out_path,'w',encoding='utf-8');
    for key in keyword:
        out_doc.write(key[0].lower());
        out_doc.write("\n");

    

path = 'dev/';
window_size_list = [2,6,10,14,20];
for r,d,f in os.walk(path):
    for file in f:
        if '.txt' in file:
            print("PROCESSING FOR FILE: ",file);
            for win_size in window_size_list:
                write_key_textrank(path,file.rstrip(".txt"),win_size);
            