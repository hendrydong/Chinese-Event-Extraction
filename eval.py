#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2017/11/16 下午12:54
# @Author      : Zoe
# @File        : eval.py
# @Description :  Your predictions should be saved in trigger_result.txt and argument_result.txt
#                 Each line contains {word  real_label predict_label}
import codecs


def evaluation(para):
    f = codecs.open(para+'_result.txt', 'r', 'utf8')
    result = f.readlines()
    f.close()


    TP, FP, TN, FN, type_correct, sum = 0, 0, 0, 0, 0, 0

    for word in result:
        if word.strip():
            sum += 1
            li = word.strip().split()

            if li[1] != 'O' and li[2] != 'O':

                TP += 1
                if li[1] == li[2]:
                    type_correct += 1
            if li[1] != 'O' and li[2] == 'O':
                FN += 1
            if li[1] == 'O' and li[2] != 'O':
                FP += 1
            if li[1] == 'O' and li[2] == 'O':
                TN += 1

    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    accuracy = (TP+TN)/sum
    F1 = 2 * precision * recall/(precision+recall)

    print('====='+para+' labeling result=====')
    print("type_correct: ", round(type_correct/TP, 4))
    print("precision: ",round(precision, 4))
    print("recall: ",round(recall,4))
    print("F1: ",round(F1,4))

evaluation('trigger')
evaluation('argument')
