# Automated car detection and identification system for surveillance

This repo consist of a Vehicle Surveillance system that  used for detecting the cars and identifying the model and manufacturing year. It can be useful for identifying and tracking the suspect vehicles in surveillance. 
  
## Dependencies

- Keras 2.2.2
- Tensorflow 1.8
- OpenCV 3.4
- imgaug 0.2.7
- Matplotlib 2.2.2
- Pillow 5.1.0

## Model 

Consists: 
 1. **Detection Module**: Car detection model based on YOLO-V2 using COCO dataset
 2. **Recognition Module**: Car classification using Stanford car dataset
 
## Datasets

 [COCO](http://cocodataset.org/#download) dataset is a collection of 123k image of 91 categories, in which I used only the **car** category. [Stanford car dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) consist of 16,185 images of 196 classes of cars.
 
Car category images can be extracted from COCO 2014 using this [repo](https://github.com/abhijithvnair94/Class-Extraction-from-COCO).

For training the Detection Module, I used total 12,753 images, in which 8,580 images were used for tarining purpose and rest for testing. For Recognition Module, I used Stanford car dataset with a split of 50-50 is used for train-test split i.e., 8,144 training images and 8,041 testing images. All the categories is mentioned in `model_names.csv`

## Pretrained model

`model.15-0.87.hdf5` - my trained weigths for recognition model

`resnet152_weights_tf.h5` - pretrained resnet-152 

`weight_coco_car_1.h5` - my trained weights for detection model 

`yolo.weight` - pretained yolo weight on COCO

Model weights mentioned above can be found [here](http://drive.google.com/open?id=12ZT_hKBt3EBVUYo3RX5iLYbMK1GNferg).

## Train

### Training Detection Module

```python train_detection.py```

provided you have set nessery data paths in the [training file](https://github.com/abhijithvnair94/Car-Detection-Model/blob/master/train_detection.py) as below:
```
wt_path = 'yolo.weights'   # path to pretrained yolo model                  
train_image_folder = '/home/abhijithmtmt17/essi/car_dataset/images/car_train/' # path to training image folder
train_annot_folder = '/home/abhijithmtmt17/essi/car_dataset/annotations/car_train_annot/' # path to training annotation folder
valid_image_folder = '/home/abhijithmtmt17/essi/car_dataset/images/car_val/' # path to validation image folder
valid_annot_folder = '/home/abhijithmtmt17/essi/car_dataset/annotations/car_val_annot/' # path to validation annotation folder
```

### Training Recoginition Module

```python train_classi.py```

provided you have set nessary data paths in the [training file](https://github.com/abhijithvnair94/Car-Detection-Model/blob/master/train_classi.py) as below:
```
train_data = './data/train' # path to training images
valid_data = './data/valid' # path to validation images
```

 ## Test
 
 1. Save all the weight files downloaded from the [google drive](http://drive.google.com/open?id=12ZT_hKBt3EBVUYo3RX5iLYbMK1GNferg) in root folder
 
 2. Test model with an test image `python main_final.py --path='path/to/test_image'`
 
 Detected car,  vechile model name and year will be displayed as below: 
 
 ![Detected](https://raw.githubusercontent.com/abhijithvnair94/Car-Detection-Model/master/sample/test5_benz/detected_benz.jpg?token=Ap7FYTHy_Smdpb18pf9AYD01gjlJtn0Rks5cMz9lwA%3D%3D)
 
 _NOTE: Some car models wasnt be able to identify(model name and year) because of lack of data/less overlap in datasets used._
