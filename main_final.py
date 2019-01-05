

#### ----------> CAR DETECTION ALONG WITH CLASSIFICATION <-------------------#######


from keras.layers import Input  
import matplotlib.pyplot as plt
import keras.backend as K
import pandas as pd
from console_progressbar import ProgressBar
import numpy as np
import pickle
import argparse
import os, cv2, time
from PIL import Image
from utils_ import  decode_netout, draw_boxes, boxes_make,load_model, final_image,crop_blob
from architecture import architecture

# GPU 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# Taking the path of test image

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='Path for test image')

loc = parser.parse_args() # object containg the test image location


# Parameters

LABELS = ['car']

IMAGE_H, IMAGE_W = 416, 416 #based on the coco dataset for detection training
GRID_H,  GRID_W  = 13 , 13
BOX              = 5
CLASS            = 1
OBJ_THRESHOLD    = 0.3
NMS_THRESHOLD    = 0.3
ANCHORS          = [0.47,0.42, 1.28,1.09, 2.68,2.54, 6.17,4.34, 10.56,9.29]

TRUE_BOX_BUFFER  = 50


# Model Architecture for YOLO


input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))

model = architecture(input_image,true_boxes)



model.load_weights("weights_coco_car_1.h5") #weight for car detection --> pretrained weight 
                                            # using COCO car dataset


path_loc = loc.path

imag_val=[]
image_name = []

for file in os.listdir(path_loc):
    if file.endswith('.jpg'):
        cc = os.path.join(file)
        i_n = file.strip('.jpg') ###
        image_name.append(i_n) ####
        imag_val.append(cc)


for m in range(len(imag_val)):
    
    image = cv2.imread(path_loc + imag_val[m])
    dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))


    input_image = cv2.resize(image, (416, 416))
    input_image = input_image / 255.
    input_image = input_image[:,:,::-1]
    input_image = np.expand_dims(input_image, 0)

    netout = model.predict([input_image, dummy_array])

    boxes = decode_netout(netout[0], 
                      obj_threshold=OBJ_THRESHOLD,
                      nms_threshold=NMS_THRESHOLD,
                      anchors=ANCHORS, 
                      nb_class=CLASS)          #create the boxes      
    b = boxes_make(image, boxes) # taking the corresponding blobes of each image
    
    
    if len(b)>0:
        
        current_loc = os.getcwd()
        new_dir = 'save_' + image_name[m]
        os.mkdir(new_dir)  #new directory for saving the final results

        for j in range(len(b)):
              
            pat = path_loc + imag_val[m]
            save_loc = current_loc + '/' + new_dir + '/'
            crop_blob(pat, (b[j][0], b[j][1], b[j][2], b[j][3]), save_loc + str(j) + '_' + imag_val[m])



img_test=[]
path =  save_loc     
for file in os.listdir(path):
    if file.endswith('.jpg'):
        cc = os.path.join(file)
        img_test.append(cc)

# Resizing the detected images

for i in range(len(img_test)):
    
    image_ = cv2.imread(path + img_test[i])
    inputimage = cv2.resize(image_, (224,224))
    inputimage = inputimage[:,:,::-1]

    res_loc = path + img_test[i]
    plt.imsave(res_loc, inputimage[:,:])



# -------> TEST USING CAR CLASSIFICATION MODEL <------------

test_image = []

for file in os.listdir(path):
    if file.endswith('.jpg'):
        cc = os.path.join(file)
        test_image.append(cc)


model = load_model() #model trained on standford car dataset

pb = ProgressBar(total=100, prefix='Predicting test data', suffix='', decimals=3, length=50, fill='=')
num_samples = len(test_image)
i=0

# start = time.time()

out = open('result_' + image_name[m] + '.txt', 'a')
for i in range(num_samples):
    filename = os.path.join(path,test_image[i])
    bgr_img = cv2.imread(filename)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (224,224))
    rgb_img = np.expand_dims(rgb_img, 0)
    preds = model.predict(rgb_img)
    prob = np.max(preds)
    
    
    if prob > 0.4:
        #print(prob)
        class_id = np.argmax(preds)
        #name = filename.strip('/home/abhijithmtmt17/essi/CAR_DETECTIONMODEL/sample/')
        out.write(test_image[i] + '\t{}\n'.format(str(class_id + 1)))
#     pb.print_progress_bar((i + 1) * 100 / num_samples)

# end = time.time()
# seconds = end - start
# print('avg fps: {}'.format(str(num_samples / seconds)))

out.close()
K.clear_session()



with open('result_' + image_name[m] + '.txt', 'r') as f:
    pred_class = [line.strip() for line in f] #predclass have category no.

imagename = [i.split('\t', 1)[0] for i in pred_class]
cat_name = [i.split('\t', 1)[1] for i in pred_class]


# Importing the classified models of Stansford car dataset

model_name = pd.read_csv('model_names.csv')
model_name_list=[]

for i in range(len(model_name)):
    
    model_name_list.append(model_name.iloc[i].values)



box_no = []
model = []
for x in range(len(cat_name)):

    cat = int(cat_name[x]) #converting the class no to integer
    image = imagename[x] 
    
    for i in range(len(model_name)):
        
        if cat == (i+1):

#             print(image + ' - ' + cat_name[x] + ' --> '+ model_name_list[i]) #
            k = image.rsplit('_',2)
            box_no.append(k[0])
            model.append(model_name_list[i])
            
        

boxes_list = [] #taking corresponding blob parameters

for val in box_no:
    y = int(val)
    boxes_list.append(b[y])


# -----------> Visualisation <----------

   # printing final image

image = cv2.imread(path_loc + imag_val[m])
plt.figure(figsize=(10,10))
image = final_image(image, boxes_list, model)
plt.imshow(image[:,:,::-1]); plt.show()
final_save_loc = save_loc + image_name[m] 
plt.imsave(final_save_loc, image[:,:,::-1])

