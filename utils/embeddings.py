# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
import shelve
import sys
import os

max_caption_length = 25
def textdata(emb,dataset,type_feat,lang,vecdb):
	s=shelve.open(vecdb)
	unique=[]
        with open('../vocabulary.txt') as f:
                for line in f:
                        unique.append(line.strip('\n'))
        vocab_size = len(unique)
        #unique = list(set(s.keys()))
        #vocab_size = len(unique)
	print(vocab_size)
	word_index = {}
	index_word = {}
	for i,word in enumerate(unique):
	    word_index[word] = i
	    index_word[i] = word
	embedding_matrix = np.zeros((vocab_size,emb))
	#embedding_matrix = np.random.uniform((vocab_size,emb))
	count=0
        for word, i in word_index.items():
		try:
	            embedding_vector = s[word]
        	    if embedding_vector is not None:
                	# words not found in embedding index will be all-zeros.
	                embedding_matrix[i] = embedding_vector
		except:
			count=count+1
			print(word)
			pass
        s.close()
	print(count)
	np.save('../seqstoretraindata/'+dataset+'/embedding_matrix.512.'+lang+'.npy',embedding_matrix)
def main():
	#emb=256
	emb=512
	lang='en'
	dataset_coco='rmeightmscoco'
	type_vgg = 'vgg16'
	type_res = 'resnet50'
	#vecdb="/data2/u1031879/kga/shelvedbs/rmeightmscoco/256.en.rmeightmscoco.db"
	vecdb="/data2/u1031879/captiongen/shelvedbs/karpathycoco/glove.512.en.karpathycoco.db"
	textdata(emb,dataset_coco,type_vgg,lang,vecdb)
if __name__=='__main__':
	main()
