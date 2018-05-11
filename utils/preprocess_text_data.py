# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
import shelve
import sys
import os

max_caption_length = 25
def imagedata(dataset,type_feat):
        data_1 = np.load('/data2/u1031879/kga/image_data/cocoattribval2014/vgg16/val_vgg16.npy')
        return data_1
def getvocab():
	unique=[]
	with open('../vocabulary.txt') as f:
		for line in f:
			unique.append(line.strip('\n'))	
	vocab_size = len(unique)
	return vocab_size,unique
def textdata(fullist,data_1,dataset,type_feat,lang):
	vocab_size,unique=getvocab()
	word_index = {}
	index_word = {}
	for i,word in enumerate(unique):
	    word_index[word] = i
	    index_word[i] = word
	num=1
	writefile = open("labels.5.en.val.txt","w")
	#writefile1 = open("captions.1.en.train.txt","w")
	for en_train in fullist:
		captions = []
		next_words=[]
		im_vecs=[]
		count=0
		with open(en_train) as f_train:
			for line in f_train:
				text = line.strip('\n').strip()
				partial_captions=[]
				nextwords=[]
				txt = text.split()
				for i in txt[1:]:
					nextwords.append(i)
				j=""
				for i in txt:
					j=j+" "+i
					partial_captions.append(j.strip())
				partial_captions.pop()
				for t,z in zip(partial_captions,nextwords):
					one = []
					for word in t.split():
						if word in word_index.keys():
							one.append(word_index[word])
						else:
							one.append(word_index['unk'])
					captions.append(one)
					if z in word_index.keys():
						next_words.append(word_index[z])
					else:
						next_words.append(word_index['unk'])
					#im_vecs.append(data_1.transpose()[count])
				count=count+1		
		#next_words_categorical = np.zeros((len(next_words),vocab_size))
		#iter1=0
		#for i in next_words:
		#	next_words_categorical[iter1,i]=1
			#writefile.write(" ".join(map(str, next_words_categorical[iter1]))+"\n")
		#	iter1=iter1+1
		
		next_words_categorical = np.asarray(next_words)
		#captions = sequence.pad_sequences(captions, maxlen=max_caption_length,padding='post')
		for val in next_words_categorical:
                         writefile.write(str(val)+"\n")
                #for val in captions:
		#	writefile.write(" ".join(map(str, val))+"\n")
		#mod = np.asarray(captions)
		#imas = np.asarray(im_vecs)
		#np.save('./storetraindata/'+dataset+'/captions.'+str(num)+'.'+lang+'.npy',mod)
		#np.save('./storetraindata/'+dataset+'/labels.'+str(num)+'.'+lang+'.npy',next_words_categorical)
		#np.save('./storetraindata/'+dataset+'/'+type_feat+'/images.'+str(num)+'.'+lang+'.npy',imas)
		#np.save('./storetraindata/'+dataset+'/vocabsize.'+lang+'.npy',vocab_size)
		num=num+1
	writefile.close()
		#print(imas.shape,mod.shape,next_words_categorical.shape)
def main():
	#emb=256
	lang='en'
	dataset_coco='rmeightmscoco'
	type_vgg = 'vgg16'
	type_res = 'resnet50'
	data_1=imagedata(dataset_coco,type_vgg)
	#en_train="./raw_data/"+dataset_coco+'/train2014/'
	#filelist=os.listdir(en_train)
	fulltrainpath=['../raw_data/rmeightmscoco/val2014/val.en.5']
	#for files in filelist:
	#	fulltrainpath.append(en_train+files)
	print(fulltrainpath)
	#vecdb = "./shelvedbs/"+dataset_iaprtc+"/"+str(i)+"."+j+"."+dataset_iaprtc+".db"
	textdata(fulltrainpath,data_1,dataset_coco,type_vgg,lang)
if __name__=='__main__':
	main()
