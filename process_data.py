#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 15:23:40 2018

process TAC40 data

@author: llq
"""

"""
Get the data file:
    (class_label e1_start e1_end e2_start e2_end sentence) 
"""
def dataset():
    train_file="./TAC40_data/test_sf3.txt"
    train_save="./TAC40_data/test_TAC40.txt"
    
    save_file=open(train_save,"w")
    with open(train_file,"r") as f:
        for lines in f.readlines():
            sentence=lines.strip().split("\t")
            #class label
            class_label=sentence[0]
            if((int)(class_label)>26):
                class_label=(int)(class_label)-1
    
            word_list=sentence[1].split(" ")
            e1_start=word_list.index("<e1>")
            e1_end=word_list.index("<\e1>")
            e1=word_list[e1_start+1:e1_end]
    
            e2_start=word_list.index("<e2>")
            e2_end=word_list.index("<\e2>")
            e2=word_list[e2_start+1:e2_end]
    
            #delete the four words
            word_list.remove("<e1>")
            word_list.remove("<\e1>")
            word_list.remove("<e2>")
            word_list.remove("<\e2>")
            
            #entity position
            e1_start=word_list.index(e1[0])
            e1_end=word_list.index(e1[-1])
            e2_start=word_list.index(e2[0])
            e2_end=word_list.index(e2[-1])
            
            word_list=" ".join(word_list)
            save_file.write(str(class_label)+" "+str(e1_start)+" "+str(e1_end)+" "+\
                str(e2_start)+" "+str(e2_end)+" "+word_list+"\n")
        
    save_file.close()

def embeddings():
    dict_file="./TAC40_data/embedding/senna/glove_6B_300vec_kbp.txt"
    word_list=open("./TAC40_data/embedding/senna/words_TAC40.lst","w")
    embeddings=open("./TAC40_data/embedding/senna/embeddings.txt","w")
    
    i=0
    with open(dict_file,"r") as f:
        for lines in f.readlines():
            i+=1
            if(i==1):
               continue
            sen_list=lines.strip().split(" ")
            #save word
            word_list.write(sen_list[0]+"\n")
            #save embeddings
            sen_list=" ".join(sen_list[1:])
            embeddings.write(sen_list+"\n")
            
            
    word_list.close()
    embeddings.close()     
    
    
dataset()
#embeddings()