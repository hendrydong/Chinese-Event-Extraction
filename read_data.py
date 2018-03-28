'''
        Copyright (c) 2017 Hanze Dong. All Rights Reserved.


        read_data.py


        read data from :
        argument_test.txt
        argument_train.txt
        trigger_test.txt
        trigger_train.txt

        and arrange them into sentences
        record the word-trigger/argument bigram, vocabulary set





'''

import codecs
import numpy as np
import codecs
def read_data():
    f = codecs.open('argument_test.txt', 'r', 'utf8')
    argument_test = f.readlines()
    f.close()
    vocab_arg = set()
    vocab_tri = set()
    test_arg_sent = [[]]
    for i in range(len(argument_test)):
        tmp = argument_test[i].strip().split('\t')
        #if tmp[0].isdecimal():tmp[0] = '0'
        if tmp[0] == '':
            vocab_arg=vocab_arg.union(set(test_arg_sent[-1]))
            test_arg_sent.append([])
            continue
        test_arg_sent[-1].append(tmp[0])




    f = codecs.open('argument_train.txt', 'r', 'utf8')
    argument_train = f.readlines()
    f.close()

    argument_tuple = []
    train_arg_sent = [[]]
    train_argument = [[]]
    for i in range(len(argument_train)):
        tmp = argument_train[i].strip().split('\t')
        #if tmp[0].isdecimal(): tmp[0] = '0'
        if tmp[0] == '':

            vocab_arg=vocab_arg.union(set(train_arg_sent[-1]))
            train_arg_sent.append([])
            train_argument.append([])
            continue
        train_arg_sent[-1].append(tmp[0])
        argument_tuple.append(tuple(tmp))
        train_argument[-1].append(tmp[1])

    f = codecs.open('trigger_train.txt', 'r', 'utf8')
    trigger_train = f.readlines()
    f.close()
    train_tri_sent = [[]]
    train_trigger = [[]]
    trigger_tuple = []
    for i in range(len(trigger_train)):
        tmp = trigger_train[i].strip().split('\t')
        #if tmp[0].isdecimal(): tmp[0] = '0'

        if tmp[0] == '':

            vocab_tri=vocab_tri.union(set(train_tri_sent[-1]))
            train_trigger.append([])
            train_tri_sent.append([])
            continue
        train_tri_sent[-1].append(tmp[0])
        trigger_tuple.append(tuple(tmp))
        train_trigger[-1].append(tmp[1])

    f = codecs.open('trigger_test.txt', 'r', 'utf8')
    trigger_test = f.readlines()
    f.close()

    test_tri_sent = [[]]

    for i in range(len(trigger_test)):
        tmp = trigger_test[i].strip().split('\t')
        #if tmp[0].isdecimal(): tmp[0] = '0'

        if tmp[0] == '':
            vocab_tri=vocab_tri.union(set(test_tri_sent[-1]))
            test_tri_sent.append([])
            continue
        test_tri_sent[-1].append(tmp[0])

    set_arg = set()
    for i in train_argument:
        set_arg = set_arg.union(set(i))
    set_tri = set()
    for i in train_trigger:
        set_tri = set_tri.union(set(i))

    train_arg_sent.remove([])
    train_tri_sent.remove([])
    test_arg_sent.remove([])
    test_tri_sent.remove([])

    return train_arg_sent,train_tri_sent,test_arg_sent,test_tri_sent,train_trigger,train_argument,set_arg,set_tri,\
           argument_tuple,trigger_tuple,vocab_arg,vocab_tri

