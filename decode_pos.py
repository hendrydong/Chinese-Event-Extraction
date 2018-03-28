'''

	Copyright (c) 2017 Hanze Dong. All Rights Reserved.

	decode_pos.py

	When we input POS information in CRF++
	it's not compatible with the eval.py

	so we use the decode_pos to make it compatible



'''





import codecs
def decode_pos(name):
	f = codecs.open(name+'_result_pos.txt', 'r', 'utf8')
	res = f.readlines()
	f.close()
	if len(res[0].split('\t'))>3:
		f = codecs.open(name+'_result.txt', 'w', 'utf8')
		for i in range(len(res)):
			tmp = res[i].split('\t')
			if len(tmp)>1:f.write(tmp[0]+'\t'+tmp[2]+'\t'+tmp[3])
			else: f.write('\n')
decode_pos('trigger')
decode_pos('argument')
