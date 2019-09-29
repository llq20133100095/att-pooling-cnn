#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
function:
    1.用来生成train和test文件格式
    2.生成words.lst
"""


def load_label2id(label2id_file):
    label2id_dict = {}
    with open(label2id_file, "r") as f:
        while True:
            line = f.readline()
            if line:
                line_list = line.strip().split(" ")
                label2id_dict[line_list[0]] = line_list[1]
            else:
                break
    return label2id_dict


def find_pos(line_data, flag):
    for index, line in enumerate(line_data.split(" ")):
        if line == flag:
            return int(index)


def process_data(train_file, train_label_file, label2id_dict, train_save):
    """
    file format:
        label + pos1_start + pos1_end + pos2_start + pos2_end + line
    :param train_file:
    :param train_label_file:
    :param label2id_dict:
    :param train_save:
    :return:
    """
    line_list = []
    with open(train_file, "r") as f:
        while True:
            line = f.readline()
            if line:
                line = line.strip()
                line_list.append(line)
            else:
                break

    label_list = []
    with open(train_label_file, "r") as f:
        while True:
            line = f.readline()
            if line:
                label_list.append(label2id_dict[line.strip()])
            else:
                break

    save_data = open(train_save, "w")
    for index, line in enumerate(line_list):
        save_data.write(label_list[index] + " ")
        e1_pos = find_pos(line, "<e1>")
        e1_pos_end = find_pos(line, "<\\e1>")
        e2_pos = find_pos(line,"<e2>")
        e2_pos_end = find_pos(line, "<\\e2>")

        if e1_pos < e2_pos:
            e2_pos -= 2
            e2_pos_end -= 2
        else:
            e1_pos -= 2
            e1_pos_end -= 2
        e1_pos_end = e1_pos + (e1_pos_end - e1_pos - 2)
        e2_pos_end = e2_pos + (e2_pos_end - e2_pos - 2)

        save_data.write(str(e1_pos) + " ")
        save_data.write(str(e1_pos_end) + " ")
        save_data.write(str(e2_pos) + " ")
        save_data.write(str(e2_pos_end) + " ")

        line = line.replace(" <e1>", "").replace(" <\\e1>", "")
        line = line.replace(" <e2>", "").replace(" <\\e2>", "")
        save_data.write(line)
        save_data.write("\n")

    save_data.close()


def get_word_lst(embedding_file, save_file, only_embedding):
    save_file = open(save_file, "w")
    only_embedding = open(only_embedding, "w")
    with open(embedding_file, "r") as f:
        f.readline()
        while True:
            line = f.readline()
            if line:
                line_list = line.split(" ")
                save_file.write(line_list[0])
                save_file.write("\n")

                only_embedding.write(" ".join(line_list[1:]))
            else:
                break
    save_file.close()
    only_embedding.close()


if __name__ == "__main__":
    """ 1. generate train and test file """
    # train_file = "../Conll04_data/raw_data/conll04_train_sen.txt"
    # train_label_file = "../Conll04_data/raw_data/conll04_train_label.txt"
    # train_save = "../Conll04_data/train_conll04.txt"
    #
    # test_file = "../Conll04_data/raw_data/conll04_test_sen.txt"
    # test_label_file = "../Conll04_data/raw_data/conll04_test_label.txt"
    # test_save = "../Conll04_data/test_conll04.txt"
    #
    # # load the label2id
    # label2id_f = "../Conll04_data/raw_data/label2id.txt"
    # label2id_dict = load_label2id(label2id_f)
    #
    # process_data(train_file, train_label_file, label2id_dict, train_save)
    # process_data(test_file, test_label_file, label2id_dict, test_save)

    """ 2. get word_list """
    embedding_conll04_file = "../Conll04_data/embedding/senna/conll04_glove_300.txt"
    list_file = "../Conll04_data/embedding/senna/words_Conll04.lst"
    only_embedding_file = "../Conll04_data/embedding/senna/embeddings.txt"
    get_word_lst(embedding_conll04_file, list_file, only_embedding_file)