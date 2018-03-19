# -*- coding: utf-8 -*-
from __future__ import print_function
import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu0")
import numpy as np
from keras.models import Model
from keras.layers import Dense, Embedding, TimeDistributed, Input, LSTM
from keras.layers.merge import Add, Concatenate
from keras.callbacks import ModelCheckpoint,Callback,History,ReduceLROnPlateau
from keras import backend as K
from keras import optimizers
def myloss(y_true,y_pred):
	"""Calculate Categorical Cross-Entropy Loss with Gradient Clipping
	"""
	y_pred = K.clip(y_pred, 0.00001, 1.0)
	return K.sparse_categorical_crossentropy(y_true, y_pred)
def languagemodel(mod,next_words_categorical,vocab_size,emb,l1,l2,mod_val,next_words_categorical_val,embedding_matrix,max_caption_length):
	"""Language Model which take word sequence as input
	"""
        writefile=open("./logs/"+"lm.training.losshistory.txt","w")
        writefile1=open("./logs/"+"lm.validation.losshistory.txt","w")
	#### Language Model ####
	embed = Embedding(vocab_size,emb,mask_zero=True,weights=[embedding_matrix],input_length=max_caption_length,trainable=True)
	lm=Input(shape=(max_caption_length,))
	lm_seq=embed(lm)
	lstm1 = LSTM(l2, return_sequences=True, recurrent_regularizer=regularizers.l2(0.00001), kernel_regularizer=regularizers.l2(0.00001))(lm_seq)
	lstm2 = LSTM(l2, return_sequences=True, recurrent_regularizer=regularizers.l2(0.00001), kernel_regularizer=regularizers.l2(0.00001))(lstm1)
	preds = TimeDistributed(Dense(vocab_size,activation='softmax'))(lstm2)
	model = Model(lm,preds)
	
	next_words_categorical_labels = np.expand_dims(next_words_categorical,axis=-1) ## expand training labels into one-hot encoding
        next_words_categorical_val_labels = np.expand_dims(next_words_categorical_val,axis=-1) ## expand validation labels into one-hot encoding
	optim = optimizers.Adam(clipnorm=1., clipvalue=0.5) ## Adam Optimizer
        model.compile(loss=myloss, optimizer=optim)
        #### End #####	

        print(model.summary()) # print model summary
        print('Number of parameters:', model.count_params()) # print model parameters
        checkpointer = ModelCheckpoint(filepath="./trainedmodels/"+"lm.weights.h5", verbose=1, save_best_only=True,monitor='val_loss')
        history=History()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=3,verbose=1, min_lr=0.00001)  ## reduce learning rate
        model.fit(mod,next_words_categorical_labels, batch_size=128, epochs=50, callbacks=[checkpointer,history,reduce_lr], validation_data=[mod_val,next_words_categorical_val_labels])

	## Write Loss History to Log files ###
        writefile.write(str(history.history['loss']))
        writefile1.write(str(history.history['val_loss']))
        writefile.close()
        writefile1.close()
def loaddata(emb,l1,l2,max_caption_length):
	"""Load unpaired text to build a language model
	"""
	## train complete MSCOCO captions 82,783 * 5 . Should be replaced with other text like Wiki, BNC copus for other unpaired text language models##
        mod_all = np.loadtxt('./seqstoretraindata/'+'mscoco.captions.train.txt', dtype="int")
        next_words_categorical_all = np.loadtxt('./seqstoretraindata/'+'mscoco.labels.train.txt', dtype="int")
        # validation MSCOC 20,242 * 5 ##
        mod_val_all=np.loadtxt('./seqstoretraindata/'+'mscoco.captions.val.txt', dtype="int")
        next_words_categorical_val_all=np.loadtxt('./seqstoretraindata/'+'mscoco.labels.val.txt', dtype="int")	
	## others ##
        embedding_matrix = np.load('./seqstoretraindata/'+'embedding_matrix.npy')
	vocab_size=embedding_matrix.shape[0]
        kgacgm(mod_all,next_words_categorical_all,int(vocab_size),emb,l1,l2,mod_val_all,next_words_categorical_val_all,embedding_matrix,max_caption_length)
def main():
        emb=256 # input word embedding dimensions
	max_caption_length = 25 # Maximum caption length
	l1,l2=512,512 # hidden layer dimensions
	loaddata(emb,l1,l2,max_caption_length)
if __name__=='__main__':
        main()
