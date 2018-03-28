'''

    Copyright (c) 2017 Hanze Dong. All Rights Reserved.

    main.py

    It's the main function to perform a HMM to extract events

    We embeded the POS information to improve the model and provide some punish options



'''





from read_data import *
import collections as cc
from HMM import *
import numpy as np
import time
from add_pos import *
import matplotlib.pyplot as plt
train_arg_sent,train_tri_sent,test_arg_sent,\
test_tri_sent,train_trigger,train_argument,set_arg,set_tri, \
argument_tuple, trigger_tuple,vocab_arg,vocab_tri=read_data()

param_arg = {'type':'arg','conf_prob':cc.Counter(argument_tuple),'arg':np.array(list(set_arg)),
             'vocab':np.array(list(vocab_arg)),'Naive_bayes':0,'k':10**5,'O_punish':0.5,'diag_punish':0.5,
             'conf_mat':np.zeros([(len(vocab_arg)),(len(set_arg))]),'k_conf_mat':10**-5,
             'lambda':5,'lambda2':2
}

param_tri = {'type':'tri','conf_prob':cc.Counter(trigger_tuple),'tri':np.array(list(set_tri)),
             'vocab':np.array(list(vocab_tri)),'Naive_bayes':0,'k':1,'O_punish':0,
             'conf_mat':np.zeros([(len(vocab_tri)),(len(set_tri))]),'k_conf_mat':10**-5,'diag_punish':0.5,
             'lambda':1,'lambda2':2
}

argument_test_pos,set_arg_te_pos = readpos('argument_test')
argument_train_pos,set_arg_tr_pos = readpos('argument_train')
trigger_test_pos,set_tri_te_pos = readpos('trigger_test')
trigger_train_pos,set_tri_tr_pos = readpos('trigger_train')
arg_pos_tuple = replace_tuple(argument_tuple,argument_train_pos)
tri_pos_tuple = replace_tuple(trigger_tuple,trigger_train_pos)
param_arg_pos = {'type':'arg','conf_prob':cc.Counter(arg_pos_tuple),'arg':np.array(list(set_arg)),
             'vocab':np.array(list(set_arg_tr_pos.union(set_arg_te_pos))),'Naive_bayes':0,'k':1,'O_punish':0,
             'conf_mat':np.zeros([len(set_arg_tr_pos.union(set_arg_te_pos)),(len(set_arg))]),'k_conf_mat':10**-5,
             'lambda':0,'diag_punish':0
}
param_tri_pos = {'type':'tri','conf_prob':cc.Counter(tri_pos_tuple),'tri':np.array(list(set_tri)),
             'vocab':np.array(list(set_tri_tr_pos.union(set_tri_te_pos))),'Naive_bayes':0,'k':1,'O_punish':0,
             'conf_mat':np.zeros([len(set_tri_tr_pos.union(set_tri_te_pos)),(len(set_tri))]),'k_conf_mat':10**-5,
             'lambda':0,'diag_punish':0.5
}
def build_conf_mat(conf_mat,conf_prob,vocab,att,add_param):
    for i in range(len(vocab)):
        for j in range(len(att)):
            conf_mat[i, j] = conf_prob[(vocab[i], att[j])]
    conf_mat += add_param
    conf_mat = conf_mat.T / np.sum(conf_mat, axis=1)
    conf_mat = conf_mat.T
    return conf_mat
t1 = time.clock()
param_arg['conf_mat']=build_conf_mat(param_arg['conf_mat'],
            param_arg['conf_prob'],param_arg['vocab'],param_arg['arg'],param_arg['k_conf_mat'])
param_arg_pos['conf_mat']=build_conf_mat(param_arg_pos['conf_mat'],
            param_arg_pos['conf_prob'],param_arg_pos['vocab'],param_arg_pos['arg'],param_arg_pos['k_conf_mat'])
param_tri['conf_mat']=build_conf_mat(param_tri['conf_mat'],
            param_tri['conf_prob'],param_tri['vocab'],param_tri['tri'],param_tri['k_conf_mat'])
param_tri_pos['conf_mat']=build_conf_mat(param_tri_pos['conf_mat'],
          param_tri_pos['conf_prob'],param_tri_pos['vocab'],param_tri_pos['tri'],param_tri_pos['k_conf_mat'])


hmm_arg = HMM(param_arg)
hmm_arg_pos = HMM(param_arg_pos)
hmm_tri = HMM(param_tri)
hmm_tri_pos = HMM(param_tri_pos)
hmm_arg.word_count(train_arg_sent)
hmm_arg_pos.word_count(argument_train_pos)
hmm_tri.word_count(train_tri_sent)
hmm_tri_pos.word_count(trigger_train_pos)
hmm_arg.trans_prob(train_argument)
hmm_arg_pos.trans_prob(train_argument)
hmm_tri.trans_prob(train_trigger)
hmm_tri_pos.trans_prob(train_trigger)




f1 = codecs.open('trigger_test.txt', 'r', 'utf8')
trigger_test = f1.readlines()
f1.close()
f2 = codecs.open('argument_test.txt', 'r', 'utf8')
argument_test = f2.readlines()
f2.close()
for i in range(len(trigger_test)):trigger_test[i]=trigger_test[i].strip()
for i in range(len(argument_test)):argument_test[i]=argument_test[i].strip()

f = codecs.open('trigger_result.txt', 'w', 'utf8')
res = []
for i in range(len(test_tri_sent)):
    x1 = hmm_tri_pos.predict(trigger_test_pos[i],'trigger')

    x = hmm_tri.predict(test_tri_sent[i], 'trigger',hmm_tri_pos.inter_mat)
    if len(set(x))==1:x = x1
    res.append(param_tri['tri'][x.astype('int')])
res2 =[]
for i in range(len(test_arg_sent)):
    x2 = hmm_arg_pos.predict(argument_test_pos[i])
    x1 = hmm_arg.predict(test_arg_sent[i],'viterbi',hmm_arg_pos.inter_mat)
    #for i in range(len(x1)):
    #   if param_arg['arg'][int(x1[i])]=='O':x1[i]=x2[i]
    res2.append(param_arg['arg'][x1.astype('int')])


t2 = time.clock()
k=0
j=0
for i in range(len(trigger_test)):
    if trigger_test[i]=='':
        f.write('\n')
        k=k+1
        j=0
    else:

        f.write(trigger_test[i])
        f.write('\t')
        f.write(res[k][j])
        f.write('\n')
        j+=1
f.close()
k=0
j=0
f = codecs.open('argument_result.txt', 'w', 'utf8')
for i in range(len(argument_test)):
    if argument_test[i]=='':
        f.write('\n')
        k=k+1
        j=0
    else:

        f.write(argument_test[i])
        f.write('\t')
        f.write(res2[k][j])
        f.write('\n')
        j+=1

#plt.imshow(hmm_arg.trans_mat)
#plt.colorbar()
#plt.show()
print('Elapsed time',t2-t1)




