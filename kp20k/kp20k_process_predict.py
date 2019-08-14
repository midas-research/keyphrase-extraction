import json
import os

def write_key_file(keyword, num):
    key_list = keyword.split(";");
    file_name = 'true_key/' + str(num) + ".key";
    if(os.path.isdir('true_key/') == 0):
        os.mkdir('true_key/');
    key_file = open(file_name,'w',encoding='utf-8');
    for key in key_list:
        key_file.write(key.lower()+"\n");
    key_file.close();
        
def write_content_file(title,abstract, num):
    file_name = "test/"+str(num)+".txt";
    if(os.path.isdir('test/') == 0):
        os.mkdir('test/');
    data_file =  open(file_name,'w',encoding='utf-8');
    
    data_file.write(title);
    data_file.write("\n");
    data_file.write(abstract);
    
    data_file.close();
    
    
open_file = open('kp20k_testing.json','r');
num = 1;

for f in open_file.readlines():
    loaded_json = json.loads(f);
    write_key_file(loaded_json['keyword'], num);
    
    write_content_file(loaded_json['title'],loaded_json['abstract'], num);
    num +=1;
    if(num==22):
        break;
    
    
    
            