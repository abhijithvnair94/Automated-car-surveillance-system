#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 16:29:33 2018

@author: Abhi
"""

import json
from pprint import pprint

with open('instances_train2014.json') as f:
    data = json.load(f)
    
data.keys()

cat = data['categories']

anno = data['annotations']

id = 3 #id for vehicle

car_list = []
for i in range(len(anno)):
    
    if (id == anno[i]['category_id']):
        
        car_list.append(anno[i])



img = data['images']

image_list=[]

for i in range(len(car_list)):
    
    img_id = car_list[i]['image_id']
    
    for j in range(len(img)):
        
        if (img_id == img[j]['id']):
            
            image = dict()
            
            a = img[j].get('file_name')
            b = img[j].get('coco_url')
            c = img[j].get('height')
            d = img[j].get('width')
            
            image["file_name"] = a
            image["coco_url"] = b
            image["height"] = c
            image["width"] = d
            
            image_list.append(image)
  

url_list = []
name_list=[]

for i in range(len(image_list)):
    
    url_list.append(image_list[i]['coco_url'])
    name_list.append(image_list[i]['file_name'])
    


import shutil
path = '/home/mtp-1/pro/essi/dataset_car/COCO/train2014/train2014/'

i_list=[]
for i in range(len(name_list)):
    
    i_list.append(i)
    shutil.copy2(path + name_list[i], '/home/mtp-1/pro/essi/dataset_car/COCO/car_train/')



#for validation data

with open('instances_val2014.json') as f:
    data_val = json.load(f)

data_val.keys()

cat_val = data_val['categories']

anno_val = data_val['annotations']

id = 3 #id for vehicle



car_list_val = []
for i in range(len(anno_val)):
    
    if (id == anno_val[i]['category_id']):
        
        car_list_val.append(anno_val[i])
        
img_val = data_val['images']



image_list_val=[]

for i in range(len(car_list_val)):
    
    img_id = car_list_val[i]['image_id']
    
    for j in range(len(img_val)):
        
        if (img_id == img_val[j]['id']):
            
            image = dict()
            
            a=0
            b=0
            c=0
            d=0
            
            a = img_val[j].get('file_name')
            b = img_val[j].get('coco_url')
            c = img_val[j].get('height')
            d = img_val[j].get('width')
            
            image["file_name"] = a
            image["coco_url"] = b
            image["height"] = c
            image["width"] = d
            
            image_list_val.append(image)




url_list_val = []
name_list_val=[]

for i in range(len(image_list_val)):
    
    url_list_val.append(image_list_val[i]['coco_url'])
    name_list_val.append(image_list_val[i]['file_name'])
 



import shutil
path = '/home/mtp-1/pro/essi/dataset_car/COCO/val2014/'

j_list=[]
for i in range(len(name_list_val)):
    
    j_list.append(i)
    shutil.copy2(path + name_list_val[i], '/home/mtp-1/pro/essi/dataset_car/COCO/car_val/')



    
    
# TAKING THE ANNOTATIONS FOR CAR TRAIN AND VAL 
    
################################################################################################    
train_xml=[]
for i in range(len(name_list)):
    
    ct = name_list[i].strip('.jpg')
    train_xml.append(ct)



import shutil
path1 = '/home/mtp-1/pro/essi/dataset_car/OUTPUT_FOLDER/train/'

j_xmllist=[]
j=0
for i in range(len(train_xml)):
    
    
    try:
        shutil.copy2(path1 + train_xml[i] + '.xml',
                 '/home/mtp-1/pro/essi/dataset_car/OUTPUT_FOLDER/car_train_annot/')

    except:
        j_xmllist.append(train_xml[i])
        j=j+1
        print(j)

with open('train_xml_miss.txt', 'w') as f:
    for item in j_xmllist:
        f.write("%s\n" % item)
####################################################################################################        
val_xml=[]
for i in range(len(name_list_val)):
    
    cv = name_list_val[i].strip('.jpg')
    val_xml.append(cv)
 
    
    
import shutil
path1 = '/home/mtp-1/pro/essi/dataset_car/OUTPUT_FOLDER/val/'

j_xmlvallist=[]
j=0
for i in range(len(val_xml)):
    
    
    try:
        shutil.copy2(path1 + val_xml[i] + '.xml',
                 '/home/mtp-1/pro/essi/dataset_car/OUTPUT_FOLDER/car_val_annot/')

    except:
        j_xmlvallist.append(val_xml[i])
        j=j+1
        print(j)
 

      
with open('val_xml_miss.txt', 'w') as f:
    for item in j_xmlvallist:
        f.write("%s\n" % item)
        
#########################################################################################################

f = open('/home/mtp-1/pro/essi/dataset_car/OUTPUT_FOLDER/val_xml_miss(7).txt', 'r+')
my_file_data = f.read().split('\n')
f.close()

val_rem=[]
for i in range(0,7):
    val_rem.append(my_file_data[i])

p='/home/mtp-1/pro/essi/dataset_car/COCO/car_val/'
for i in range(len(val_rem)):
    shutil.move(p + val_rem[i] + '.jpg' , '/home/mtp-1/pro/essi/dataset_car/OUTPUT_FOLDER/')
##############################################3

f = open('/home/mtp-1/pro/essi/dataset_car/OUTPUT_FOLDER/train_xml_miss(26).txt', 'r+')
my_file_data = f.read().split('\n')
f.close()

train_rem=[]
for i in range(0,26):
    train_rem.append(my_file_data[i])

p='/home/mtp-1/pro/essi/dataset_car/COCO/car_train/'
for i in range(len(train_rem)):
    shutil.move(p + train_rem[i] + '.jpg' , '/home/mtp-1/pro/essi/dataset_car/COCO/miss_train/')


