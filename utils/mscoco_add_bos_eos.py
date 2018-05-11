import json
from pprint import pprint
from collections import defaultdict

def parsecoco(filename):
	with open(filename) as json_data:
	    d = json.load(json_data)
	dicts_images = d['images']
	list_image={}
	for image in dicts_images:
		list_image[image["id"]]=image["file_name"]
	dicts_annot = d['annotations']
	dictid=defaultdict(str)
	for val in dicts_annot:
		if val['image_id'] in list_image.keys():
			dictid[str(val['image_id'])+"::"+list_image[val["image_id"]]]+=val['caption']+"$$"
	addbos(dictid)

def addbos(dictid):
	en_train_1="./raw_data/mscoco_bos/train2014/train.en.1"
	en_train_2="./raw_data/mscoco_bos/train2014/train.en.2"
	en_train_3="./raw_data/mscoco_bos/train2014/train.en.3"
	en_train_4="./raw_data/mscoco_bos/train2014/train.en.4"
	en_train_5="./raw_data/mscoco_bos/train2014/train.en.5"
	en_1=open(en_train_1,"w")
	en_2=open(en_train_2,"w")
	en_3=open(en_train_3,"w")
	en_4=open(en_train_4,"w")
	en_5=open(en_train_5,"w")
	for i,j in dictid.iteritems():
		sent_list=j.strip("$$").split("$$")
		num=0
		while num < 5:
			sent="BOS"
			sent1=sent_list[num].strip().replace(".","").replace(",","").split()
			for i in sent1:
				sent=sent+" "+i
			sent=sent.strip()
			sent=sent+" "+"EOS"
			if num==0:
				en_1.write(sent.lower()+"\n")
			elif num==1:
				en_2.write(sent.lower()+"\n")
			elif num==2:
				en_3.write(sent.lower()+"\n")
			elif num==3:
				en_4.write(sent.lower()+"\n")
			elif num==4:
				en_5.write(sent.lower()+"\n")
			num=num+1
	en_1.close()
	en_2.close()
	en_3.close()
	en_4.close()
	en_5.close()
def main():
	filename="./annotations/captions_train2014.json"
	parsecoco(filename)
if __name__=="__main__":
	main()
