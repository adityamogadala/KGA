# -*- coding: utf-8 -*-
from __future__ import print_function
import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu0")
import numpy as np
from keras.models import Model
from keras.layers import Dense, Embedding, TimeDistributed, Input, LSTM, RepeatVector
from keras.layers.merge import Add, Concatenate
from keras.layers.core import Permute, Activation
from keras.callbacks import ModelCheckpoint,Callback,History,ReduceLROnPlateau
from keras.engine.topology import Layer, InputSpec
from keras import backend as K
from keras import initializers,regularizers,constraints,optimizers
class SemanticAttLayer(Layer):
	"""Class which creates Semantic Attention Layer.
	"""
    def __init__(self, W_regularizer=None, W_constraint=None, **kwargs):
	"""Initialize Weight Matrix W with Glorot Uniform and 
	   without Regularization and Constraints
	"""
	self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        super(SemanticAttLayer, self).__init__(** kwargs)

    def build(self, input_shape):
	"""Setup Dimensions of Weight Matrix W without bias
	"""
	self.W = self.add_weight((input_shape[1][-1],input_shape[0][-1]),initializer=self.init,name='{}_W'.format(self.name),trainable=True)
	self.bias=None
        super(SemanticAttLayer, self).build(input_shape)
    def compute_mask(self, input, input_mask=None):
	"""Mask is not passed to next layers
	"""
	return None
    def call(self, x, mask=None):
	"""Calculate Attention Weights between Entity Labels and Generated Word Vectors
	"""
	inw = K.tanh(K.dot(x[1],self.W))
	inw1 = K.permute_dimensions(inw,(0,2,1))
	we = K.tanh(K.batch_dot(x[0],inw1))
        ap = K.exp(we-K.max(we,axis=-1,keepdims=True))
        sp= K.cast(K.sum(ap, axis=-1, keepdims=True) + 0.000001, K.floatx()) ## Add small value in denominator to avoid division by zero
	a = ap/sp
        weighted_input = K.batch_dot(a,x[1])
	return weighted_input
    def compute_output_shape(self, input_shape):
	""" Returns the final Output from the Semantic Attention Weighted Entity Label Vectors
	"""
	return (input_shape[0][0],input_shape[0][1],input_shape[1][-1])
def myloss(y_true,y_pred):
	"""Calculate Categorical Cross-Entropy Loss with Gradient Clipping
	"""
	y_pred = K.clip(y_pred, 0.00001, 1.0)
	return K.sparse_categorical_crossentropy(y_true, y_pred)
def kgacgm(imas,mod,next_words_categorical,vocab_size,dim,emb,l1,l2,imas_val,mod_val,next_words_categorical_val,embedding_matrix,sl_train,sl_val,max_caption_length):
	"""Knowledge Guided Assisted (KGA) Caption Generation Model which takes input from image features, caption word sequences and Entity labels.
	"""
        writefile=open("./logs/"+"cm.training.txt","w")
        writefile1=open("./logs/"+"cm.validation.txt","w")
	
	## Language Model ##
        embed = Embedding(vocab_size,emb,mask_zero=True,weights=[embedding_matrix],input_length=max_caption_length,trainable=True)
        lm=Input(shape=(max_caption_length,))
        lm_seq=embed(lm)
        lstm1 = LSTM(l2, return_sequences=True, recurrent_regularizer=regularizers.l2(0.00001), kernel_regularizer=regularizers.l2(0.00001))(lm_seq)
        lstm2 = LSTM(l2, return_sequences=True, recurrent_regularizer=regularizers.l2(0.00001), kernel_regularizer=regularizers.l2(0.00001))(lstm1)
        preds = TimeDistributed(Dense(vocab_size,activation='softmax'))(lstm2)
        model = Model(lm,preds)
        model.load_weights('./trainedmodels/lm.weights.h5')
	
	### Start of Caption Model  ###
	for layer in model.layers:
		layer.trainable = False
        ## Entity labels ##
	sl_input = Input(shape=(5,500))
        l_att = SemanticAttLayer()([lstm2,sl_input])
        l_att_dense = TimeDistributed(Dense(vocab_size,name='sematt'))(l_att) # for addition
        ## Image ##
	image = Input(shape=(dim,))
        image_dense1=(RepeatVector(max_caption_length))(image)
	image_dense2 = TimeDistributed(Dense(vocab_size,name='image_w'))(image_dense1)        
	## Language ##
        lstm2_dense = TimeDistributed(Dense(vocab_size,name='lm'))(model.layers[3].output)
	merge0 = [image_dense2,lstm2_dense,l_att_dense]
	merge = Add()(merge0)
        preds = TimeDistributed(Activation('softmax'))(merge)
        model_fin = Model([image,lm,sl_input],preds)
	
	next_words_categorical_labels = np.expand_dims(next_words_categorical,axis=-1)
        next_words_categorical_val_labels = np.expand_dims(next_words_categorical_val,axis=-1)
	optim = optimizers.Adam(clipnorm=1., clipvalue=0.5) ## Adam Optimizer
        model_fin.compile(loss=myloss, optimizer=optim)
	####### End of Caption Model ######################
        print(model_fin.summary())             ## print model summary
        print('Number of parameters:', model_fin.count_params()) ## print model parameters

        checkpointer = ModelCheckpoint(filepath="./trainedmodels/"+"captionmodel.weights.h5", verbose=1, save_best_only=True,monitor='val_loss') ## Save Weights of the Model
        history=History()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=3,verbose=1,min_lr=0.00001) ## Reduce Learning Rate 
        model_fin.fit([imas, mod, sl_train],next_words_categorical_labels, batch_size=128, epochs=50, callbacks=[checkpointer,history,reduce_lr], validation_data=[[imas_val,mod_val,sl_val],next_words_categorical_val_labels])
        
	## Write Loss History to Log files ###
	writefile.write(str(history.history['loss']))
        writefile1.write(str(history.history['val_loss']))
        writefile.close()
        writefile1.close()

def loaddata(dim,emb,l1,l2,max_caption_length):
	"""Load Training and Validation Data consituting Image Features, Caption Word Sequences and Entity Label Vectors
	"""
	## training size = 70,194 * 5 , MSCOCO 8 objects removed dataset##
	imas_all = np.loadtxt('./seqstoretraindata/'+'rmeightmscoco.images.cocoattrib.train.txt', dtype="float")
        mod_all = np.loadtxt('./seqstoretraindata/'+'rmeightmscoco.captions.train.txt', dtype="int")
        next_words_categorical_all = np.loadtxt('./seqstoretraindata/'+'rmeightmscoco.labels.train.txt', dtype="int")
	sl_train = np.load('./seqstoretraindata/train-top5-entity-label-vecs.npy')
        # validation size = 20,242 * 5  ##
        imas_val_all=np.loadtxt('./seqstoretraindata/'+'rmeightmscoco.images.cocoattrib.val.txt',dtype="float")
        mod_val_all=np.loadtxt('./seqstoretraindata/'+'rmeightmscoco.captions.val.txt', dtype="int")
        next_words_categorical_val_all=np.loadtxt('./seqstoretraindata/'+'rmeightmscoco.labels.val.txt', dtype="int")
	sl_val = np.load('./seqstoretraindata/val-top5-entity-label-vecs.npy')
	
        embedding_matrix = np.load('./seqstoretraindata/'+'embedding_matrix.npy')
	vocab_size=embedding_matrix.shape[0]
        kgacgm(imas_all,mod_all,next_words_categorical_all,int(vocab_size),dim,emb,l1,l2,imas_val_all,mod_val_all,next_words_categorical_val_all,embedding_matrix,sl_train,sl_val,max_caption_length)
def main():
        vgg_dim=471 ## image features dimensions
	max_caption_length = 25  # Set your Maximum Caption Length
        emb=256 ## word embedding dimensions
	l1,l2=512,512 ## hidden layer dimensions
	loaddata(vgg_dim,emb,l1,l2,max_caption_length)
if __name__=='__main__':
        main()
