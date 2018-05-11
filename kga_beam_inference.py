#from __future__ import print_function
import numpy as np
import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu1")
import copy
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Reshape, RepeatVector, Dense, Embedding, TimeDistributed, LSTM, Input
from keras.layers.merge import Add, Concatenate
from keras.layers.core import Activation
from keras.engine.topology import Layer, InputSpec
from keras import backend as K
from keras import initializers,optimizers,regularizers,constraints
import re

from scipy.spatial import distance
import math

batch_size,max_caption_length,dim,emb,l1,l2 = 1,25,471,256,512,512 # Set Initial Global Variables (caption length, image feature dimensions, word embedding dimensions, hidden layer dimensions)
dataset="rmeightmscoco"
lang="en"
iv=256
def getvocab():
	"""Fetch Caption Vocabulary
        """
        unique=[]
        with open('./vocablist/vocabulary.txt') as f:
                for line in f:
                        unique.append(line.strip('\n'))
        vocab_size = len(unique)
        word_index = {}
        index_word = {}
        for i,word in enumerate(unique):
            word_index[word] = i
            index_word[i] = word
        return word_index,vocab_size,index_word
def getattributes():
	"""Fetch Attributes for Transferring Features
        """
	allatt = './vocablist/allimagettributes.txt'
	origatt = './vocablist/origimageattributes.txt'
	tran_word = './vocablist/transfer_words_coco1.txt'
	tran_classi = './vocablist/transfer_classifiers_coco1.txt'
	all_attributes = []
	orig_attributes = []
	transfer_vocab_words = []
	transfer_classifier_words = []
	with open(allatt) as f:
		for line in f:
			all_attributes.append(line.strip('\n'))
	with open(origatt) as f:
		for line in f:
			orig_attributes.append(line.strip('\n'))
	with open(tran_word) as f:
		for line in f:
			transfer_vocab_words.append(line.strip('\n'))
	with open(tran_classi) as f:
		for line in f:
			transfer_classifier_words.append(line.strip('\n'))
	new_attributes = list(set(all_attributes)-set(orig_attributes))
	return all_attributes,orig_attributes,new_attributes,transfer_vocab_words,transfer_classifier_words
def getkblabels():
	"""Get Top5 Knowledge Graph Labels
        """
	testkb = './vocablist/test_kb_top5_labels.txt'
	testkb_labels=[]
	with open(testkb) as f:
                for line in f:
                        testkb_labels.append(line.strip('\n'))
	return testkb_labels

################### Model-Start ############################
word_index,vocab_size,index_word=getvocab()
allattri,origattri,newattri,transwords,transclassi = getattributes()
testkblabels=getkblabels()
embedding_matrix = np.load('./seqstoretraindata/'+'embedding_matrix.npy')
##################### Caption Model - Start  ####################################################
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
        super(SemanticAttLayer, self).build(input_shape)  # be sure you call this somewhere!
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
        ap = K.exp(we)
        sp = K.cast(K.sum(ap, axis=-1, keepdims=True) + 0.000001, K.floatx())
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
def captionmodel():
	"""Knowledge Guided Assisted (KGA) Caption Generation Model which takes input from image features, caption word sequences and Entity labels.
        """
	## LM ##
	embed = Embedding(vocab_size,emb,mask_zero=True,weights=[embedding_matrix],input_length=max_caption_length,trainable=True)
        lm=Input(shape=(max_caption_length,))
        lm_seq=embed(lm)
        lstm1 = LSTM(l2, return_sequences=True, recurrent_regularizer=regularizers.l2(0.00001), kernel_regularizer=regularizers.l2(0.00001))(lm_seq)
        lstm2 = LSTM(l2, return_sequences=True, recurrent_regularizer=regularizers.l2(0.00001), kernel_regularizer=regularizers.l2(0.00001))(lstm1)
        ## Semantic Attention ##
        sl_input = Input(shape=(5,500))
        l_att = SemanticAttLayer()([lstm2,sl_input])
        l_att_dense = TimeDistributed(Dense(8802,name='sematt'))(l_att) # for addition
        ## Image ##
        image = Input(shape=(dim,))
        image_dense1=(RepeatVector(max_caption_length))(image)
        image_dense2 = TimeDistributed(Dense(8802,name='image_w'))(image_dense1)

        lstm2_dense = TimeDistributed(Dense(8802,name='lm'))(lstm2)
        merge0 = [image_dense2,lstm2_dense,l_att_dense]
        merge = Add()(merge0)
        preds = TimeDistributed(Activation('softmax'))(merge)
        model1 = Model([image,lm,sl_input],preds)
	model1.load_weights('./trainedmodels/captionmodel.weights.h5')
	return model1
def transferedmodel():
	"""Modifies Caption Generation Model weights with seen words in captions with unseen visual object categories
        """
	model=captionmodel()
	closeword={}
	reversecloseword={}
	closeword_f,closword_f_ext = './vocablist/closeword.txt', './vocablist/closeword_ext_b.txt' # Removed 8 Objects Dataset MSCOCO. Change for ImageNet.
        with open(closeword_f) as f:
                for line in f:
                        items=line.strip('\n').split('$')
                        closeword[items[0]]=items[1]
        with open(closeword_f_ext) as f:
                for line in f:
                        items=line.strip('\n').split('$')
                        reversecloseword[items[1]]=items[0]
        #### Updated Caption Model - Start ################
        ### Transfer weights between seen words and visual object categories ##
	t=copy.deepcopy(model.layers[8].get_weights())
        t_lm=copy.deepcopy(model.layers[9].get_weights())
        t_sa=copy.deepcopy(model.layers[10].get_weights())
        for word,word1 in zip(transclassi,transwords):
                word_idx = word_index[word1]
                transfer_word_idx = word_index[closeword[word1]]
                attribute_idx=allattri.index(word)
                if word1=='zebras':
                        transfer_attribute_idx = allattri.index('giraffe')
                else:
                        transfer_attribute_idx = allattri.index(closeword[word1])
                transfer_weights_im = np.array([0.0]*471)
                transfer_weights_lm = np.array([0.0]*512)
                transfer_weights_sa = np.array([0.0]*500)
                transfer_weights_im += t[0].transpose()[transfer_word_idx,:]
                transfer_weights_lm += t_lm[0].transpose()[transfer_word_idx,:]
                transfer_weights_sa += t_sa[0].transpose()[transfer_word_idx,:]
                transfer_weights_im[attribute_idx] = t[0].transpose()[transfer_word_idx, transfer_attribute_idx]
                transfer_weights_im[transfer_attribute_idx] = 0
                t[0].transpose()[word_idx,:] = transfer_weights_im
                t_lm[0].transpose()[word_idx,:] = transfer_weights_lm
                t_sa[0].transpose()[word_idx,:] = transfer_weights_sa
                t[0].transpose()[transfer_word_idx,attribute_idx] = 0
        model.layers[8].set_weights(t)
        model.layers[9].set_weights(t_lm)
        model.layers[10].set_weights(t_sa)
	#### Updated Caption Model - End ################
        optim = optimizers.Adam(clipnorm=1., clipvalue=0.5)
        model.compile(loss=myloss, optimizer=optim)
	return model,closeword,reversecloseword
################# Model-End ##################################
model,closeword,reversecloseword = transferedmodel()
def LSTM_predict_first_step(image_data,labels,captions,vocab_size,embedding_matrix):
        value = model.predict([image_data,captions,labels],batch_size=1,verbose=0)[0]
        return value,closeword,reversecloseword
def beam_search(image_data,labels,sentence_candidates,final_sentences,depth,beamsize,vocab_size,embedding_matrix,word_index,index_word):
#        volatile=True
        next_sentence_candidates_temp=list()
        for sentence_tuple in sentence_candidates:
                cur_sentence = sentence_tuple[0]
                cur_log_likely=sentence_tuple[1]
                captions=[]
                captions.append(cur_sentence)
                captions = sequence.pad_sequences(captions, maxlen=max_caption_length,padding='post')
                predicted_word_np,closeword,reversecloseword = LSTM_predict_first_step(image_data,labels,captions,vocab_size,embedding_matrix)
		top_indexes = (-predicted_word_np[depth,:]).argsort()[:beamsize]
                for index in np.nditer(top_indexes):
                        index=int(index)
        		word_pred = index_word[index] # check for transfer
			if word_pred in reversecloseword.keys():
				index=word_index[reversecloseword[word_pred]] # word transfer
        	        	probability=predicted_word_np[depth,index]
                	      	next_sentence = copy.deepcopy(cur_sentence)
	                	next_sentence.append(index)
        	              	log_likely=math.log(probability)
	                	next_log_likely=cur_log_likely+log_likely
        	                next_sentence_candidates_temp.append((next_sentence,next_log_likely))# make each sentence tuple
                        else:
				probability=predicted_word_np[depth,index]
        	                next_sentence = copy.deepcopy(cur_sentence)
                	        next_sentence.append(index)
                        	log_likely=math.log(probability)
	                        next_log_likely=cur_log_likely+log_likely
        	                next_sentence_candidates_temp.append((next_sentence,next_log_likely))# make each sentence tuple
        prob_np_array=np.array([sentence_tuple[1] for sentence_tuple in next_sentence_candidates_temp])
        top_candidates_indexes=(-prob_np_array).argsort()[:beamsize]
        next_sentence_candidates=list()

        for i in top_candidates_indexes:
                sentence_tuple=next_sentence_candidates_temp[i]
                index=sentence_tuple[0][-1]
		#print index_word[index]
                if index_word[index]=='eos':
                        final_sentence=sentence_tuple[0]
                        final_likely=sentence_tuple[1]
                        final_probability=math.exp(final_likely)
                        final_sentences.append((final_sentence,final_probability,final_likely))
                else:
                        next_sentence_candidates.append(sentence_tuple)
        if len(final_sentences)>=1:
            return final_sentences
        elif depth==(max_caption_length-1):
            return final_sentences
        else:
            depth+=1
            return beam_search(image_data,labels,next_sentence_candidates,final_sentences,depth,beamsize,vocab_size,embedding_matrix,word_index,index_word)
def main():
	data_path='./seqstoretraindata/'
        tim_ind='./test_im_indexes/'
	data_1 = np.loadtxt(data_path+'rmeightmscoco.images.cocoattrib.test.txt', dtype='float') # Test Images Features
        data_2 = np.load(data_path+'test-top5-entity-label-vecs.npy') # Test Entity Label Vectors
	files = [tim_ind+'bottle_im_index.txt',tim_ind+'bus_im_index.txt',tim_ind+'couch_im_index.txt',tim_ind+'microwave_im_index.txt',tim_ind+'pizza_im_index.txt',tim_ind+'racket_im_index.txt',tim_ind+'suitcase_im_index.txt',tim_ind+'zebra_im_index.txt'] # Test MSCOCO Images Index
        obi=['bottle','bus','couch','microwave','pizza','racket','suitcase','zebra'] # MSCOCO Objects, replace with ImageNet Objects
	for objs,fili in zip(obi,files):
		im_ind=[]
		with open(fili) as f:
        		for line in f:
                		im_ind.append(int(line.strip('\n')))
		depth=0
		beamsize=3
		count=0
		mycount=0
		filename='./gensentences/unseen_'+objs+'_generated_sentences.txt'
		writefile=open(filename,'w')
		for im,orgsent,labels,kb_labels in zip(data_1,org_sent,data_2,testkblabels):
			if count in im_ind:
				probs=1.0
		        	initial_sent_cand = [([word_index['bos']],probs)]
				final_sent=list()
				generated_sentences=beam_search(np.asarray(im).reshape(1,dim),labels.reshape(1,5,500),initial_sent_cand,final_sent,depth,beamsize,vocab_size,embedding_matrix,word_index,index_word)
				if len(generated_sentences)>0:
                        		sentence=[index_word[index] for index in generated_sentences[0][0]][1:-1]
	        	                writefile.write(' '.join(sentence)+'\n')
				else:
	        	                writefile.write('a'+'\n')
				mycount=mycount+1
			count=count+1
		writefile.close()
if __name__=='__main__':
	main()
