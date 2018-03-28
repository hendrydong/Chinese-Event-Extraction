'''
    Copyright (c) 2017 Hanze Dong. All Rights Reserved.

    add_pos.py

    Return the POS information of the dataset.



'''

from jieba.posseg import lcut

import codecs
import itertools
def pos(name):
    f = codecs.open(name+'.txt', 'r', 'utf8')
    data = f.readlines()
    f.close()
    g = codecs.open(name+'_pos.txt', 'w', 'utf8')
    data_pos = data
    for i in range(len(data)):
        if not data[i]=='\n':
            data[i] = data[i].split('\t')[0].strip()
            data_pos[i] = list(lcut(data[i])[-1])[1]
            g.write(data_pos[i])
            g.write('\n')
        else:g.write('\n')


    g.close()
def readpos(name):
    f = codecs.open(name+'_pos.txt', 'r', 'utf8')
    res  = f.readlines()
    res_sent = [[]]
    for i in range(len(res)):
        tmp = res[i].strip()
        if tmp == '':
            res_sent.append([])
            continue
        res_sent[-1].append(tmp)
    set_pos = set()
    res_sent.remove([])
    for i in res_sent:
        set_pos = set_pos.union(set(i))
    return res_sent,set_pos

def replace_tuple(list_tuple0,pos_sent):
    pos_sent = list(itertools.chain.from_iterable(pos_sent))
    res_tuple = list_tuple0
    for i in range(len(list_tuple0)):
        res_tuple[i] = (pos_sent[i],list_tuple0[i][1])
    return res_tuple

def write_pos(name):
    f = codecs.open(name+'_pos.txt', 'r', 'utf8')
    pos  = f.readlines()
    g = codecs.open(name+'.txt', 'r', 'utf8')
    data = g.readlines()
    h = codecs.open(name+'2.txt', 'w', 'utf8')
    for i in range(len(pos)):
        if pos[i]=='\n':h.write('\n')
        else:
            tmp = data[i].strip().split('\t')

            h.write(tmp[0]+'\t'+pos[i].strip()+'\t'+tmp[1]+'\n')

    

if __name__=='__main__':
    pos('argument_test')
    pos('argument_train')
    pos('trigger_train')
    pos('trigger_test')
    write_pos('argument_test')
    write_pos('argument_train')
    write_pos('trigger_train')
    write_pos('trigger_test')

