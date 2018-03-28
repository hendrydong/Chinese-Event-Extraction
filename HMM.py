'''
Copyright (c) 2017 Hanze Dong. All Rights Reserved.
	HMM.py

	HMM class include the main part of Hidden Markov Model


	We provide 2 types of optimization :
		Viterbi and Trigger(which is designed for trigger detection)



'''
import nltk
import collections as cc
import numpy as np
class HMM(object):
	"""docstring for ClassName"""
	def __init__(self, param):
		self.param = param

	def word_count(self,sent):
		self.count_word = cc.Counter()
		for i in range(len(sent)):
			self.count_word = cc.Counter(sent[i])


	def trans_prob(self,hidden_chain):
		self.ini_cond = cc.Counter()
		self.count_hidden_cond = cc.Counter()
		for i in range(len(hidden_chain)):
			self.ini_cond+=cc.Counter(hidden_chain[i])
			self.count_hidden_cond+=cc.Counter(list(nltk.ngrams(hidden_chain[i],2)))
		self.trans_mat = np.zeros([(len(self.param[self.param['type']])),
								   (len(self.param[self.param['type']]))])
		self.prob_hidden = np.zeros((len(self.param[self.param['type']])))
		for i in range(len(self.param[self.param['type']])):
			self.prob_hidden[i] = self.ini_cond[self.param[self.param['type']][i]]
			for j in range(len(self.param[self.param['type']])):
				self.trans_mat[i,j]=\
					self.count_hidden_cond[(self.param[self.param['type']][i],self.param[self.param['type']][j])]
		self.prob_hidden = self.prob_hidden/np.sum(self.prob_hidden)
		O_idx = np.argwhere(self.param[self.param['type']]=='O')
		self.O_idx = O_idx
		self.trans_mat+=self.param['k']
		self.trans_mat[:,O_idx] = self.param['O_punish']*self.trans_mat[:,O_idx]
		self.trans_mat =self.trans_mat - (1-self.param['diag_punish'])*np.diag(np.diag(self.trans_mat))
		self.trans_mat = (self.trans_mat.T/np.sum(self.trans_mat,axis=1)).T

		if self.param['Naive_bayes']:
			self.trans_mat = np.ones([len(self.trans_mat),len(self.trans_mat)])


	def predict(self,sent,mode='viterbi',inter_mat=[]):
		word_idx = np.zeros((len(sent)))
		for i in range(len(sent)):
			word_idx[i]=int(np.argwhere(self.param['vocab']==sent[i]).flatten())
		word_idx=word_idx.astype('int')
		emis_mat = self.param['conf_mat'][word_idx,:]
		trans_mat = self.trans_mat
		prob_hidden = self.prob_hidden
		self.inter_mat = emis_mat
		if mode == 'trigger':
			x = self.trigger(sent, emis_mat, trans_mat, prob_hidden,inter_mat)
		elif mode == 'viterbi':
			x = self.viterbi(sent, emis_mat, trans_mat, prob_hidden,inter_mat)

		return x

	def greedy(self,sent,emis_mat,trans_mat,prob_hidden):
		pass


	def viterbi(self,sent,emis_mat,trans_mat,prob_hidden,inter_mat):
		t1 = np.zeros([len(trans_mat),len(sent)])
		t2 = np.zeros([len(trans_mat),len(sent)])
		t1[:,0] = emis_mat[0,:]*prob_hidden+self.param['lambda']* emis_mat[0,:]

		x = np.zeros(len(sent))
		for i in range(1,len(sent)):
			tmp = (t1[:,i-1] * trans_mat.T)
			if len(inter_mat)==0:tmp2 = (tmp.T+self.param['lambda']) * emis_mat[i,:]
			else:tmp2 = (tmp.T+self.param['lambda']) * emis_mat[i,:]+self.param['lambda2']*inter_mat[i,:]
			t1[:, i] = np.max(tmp2,axis=0)
			t2[:, i] = np.argmax(tmp2,axis=0)

		x[-1] = self.new_arg_max(t1[:, len(sent)-1],self.O_idx)
		for i in range(2,len(sent)+1):
			x[len(sent)-i]=t2[int(x[len(sent)-i+1]),len(sent)-i+1]
		return x

	def trigger(self,sent,emis_mat,trans_mat,prob_hidden,inter_mat):
		x = np.zeros(len(sent))
		t1 = np.zeros([len(trans_mat), len(sent)])
		O_inx = np.argwhere(self.param['tri']=='O')
		x += O_inx[0]

		for i in range(len(sent)):
			if len(inter_mat) == 0:t1[:,i] = trans_mat[O_inx,:]*emis_mat[i,:]*trans_mat[:,O_inx].T+\
											 self.param['lambda']*emis_mat[i,:]
			else:t1[:,i] = trans_mat[O_inx,:]*emis_mat[i,:]*trans_mat[:,O_inx].T+\
						   self.param['lambda']*emis_mat[i,:]+\
						   self.param['lambda2']*inter_mat[i,:]
		t1[O_inx,:] = 0
		t2 = np.max(t1,axis=0)
		t3 = np.argmax(t1,axis=0)
		if np.max(t2)>np.min(t2):
			x[np.argmax(t2).astype('int')]=t3[np.argmax(t2).astype('int')]
		return x

	def new_arg_max(self,x,default=0):
		if min(x)==max(x):return default
		else:return np.argmax(x)










		
		