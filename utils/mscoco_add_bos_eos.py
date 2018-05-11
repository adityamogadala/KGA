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
	#en_train_images="./raw_data/mscoco/test2014/test_images.txt"
	en_1=open(en_train_1,"w")
	en_2=open(en_train_2,"w")
	en_3=open(en_train_3,"w")
	en_4=open(en_train_4,"w")
	en_5=open(en_train_5,"w")
	#en_images=open(en_train_images,"w")
	for i,j in dictid.iteritems():
	#	en_images.write(i.split("::")[1].strip()+"\n")
		sent_list=j.strip("$$").split("$$")
		num=0
		while num < 5:
			sent="BOS"
			sent1=sent_list[num].strip().replace(".","").replace(",","").split()
			for i in sent1:
				sent=sent+" "+i
			sent=sent.strip()
			sent=sent+" "+"EOS"
        		#lst = sent.split()
			#cap = lst[1][0].upper()
			#lst[1] = cap+lst[1][1:]
			#sent = ' '.join(lst)
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
	#en_images.close()
def main():
	#filename="/localdata/u1031879/DCC/annotations/captions_no_caption_rm_eightCluster_train2014.json"
	filename="/localdata/u1031879/DCC/annotations/captions_train2014.json"
	#filename="/localdata/u1031879/DCC/annotations/captions_val_test2014.json"
	parsecoco(filename)
if __name__=="__main__":
	main()
