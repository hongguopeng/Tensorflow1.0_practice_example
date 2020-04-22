#### 請解壓縮data.rar，取得本程式之數據 #####
#生成tf文件
import tensorflow as tf
import os
import random
import sys
import numpy as np
import cv2


# 數據集路徑
DATASET_DIR = os.path.join('captcha' , 'images')

# tfrecord文件存放路徑
TFRECORD_DIR = os.path.join('captcha')

# 讀取取所有image
photo_filenames = []
for filename in os.listdir(DATASET_DIR):
    path = os.path.join(DATASET_DIR , filename) #文件路徑
    photo_filenames.append(path)

# 劃分驗證集訓練集
num_test = 300
# 切分數據為測試集和訓練集，並打亂
random.seed(0)
random.shuffle(photo_filenames)
training_filenames = photo_filenames[num_test:]
testing_filenames = photo_filenames[:num_test]


def int64_feature(values):
    if not isinstance(values , (tuple , list)) : 
        values = [values]
    return tf.train.Feature(int64_list = tf.train.Int64List(value = values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

# image轉換成tfexample函數
def image_to_tfexample(image_data , label0 , label1 , label2 , label3):
    return tf.train.Example(features =
                            tf.train.Features(feature =  {'image': bytes_feature(image_data) ,
                                                          'label0': int64_feature(label0) ,
                                                          'label1': int64_feature(label1) ,
                                                          'label2': int64_feature(label2) ,
                                                          'label3': int64_feature(label3) } ))

# 數據轉換城tfrecorad格式
def convert_dataset(split_name , filenames):

    # 定義tfrecord的路徑名字
    output_filename = os.path.join(TFRECORD_DIR , split_name + '.tfrecords')
    
    # 判斷tfrecord文件是否存在，若存在則把原來tfrecord文件刪除
    tfrecords_name = split_name + '.tfrecords'
    if tfrecords_name in os.listdir(TFRECORD_DIR):
        os.remove(output_filename)
    
    # 生成tfrecords文件
    tfrecord_writer = tf.python_io.TFRecordWriter(output_filename) 

    for i , filename in enumerate(filenames):
        try:
            sys.stdout.write('\r>> Converting image : {}/{}'.format(i + 1 , len(filenames) ))
            sys.stdout.flush()
            
            # 讀取image   
            image_data = cv2.imread(os.path.join(filename))
            
            # 對image進行resize
            image_data = cv2.resize(image_data , (224 , 224) , interpolation = cv2.INTER_CUBIC)
            
            # 灰度化
            image_data = cv2.cvtColor(image_data , cv2.COLOR_BGR2GRAY)
            
            # 將image轉為bytes
            image_data = image_data.tobytes()
            
            # 獲取label
            labels = filename[-8 : -4]
            num_labels = [int(labels[j]) for j in range(0 , 4)]
    
    
            # 生成tfrecord文件
            example = image_to_tfexample(image_data , 
                                         num_labels[0] ,
                                         num_labels[1] , 
                                         num_labels[2] , 
                                         num_labels[3])
            # 寫入數據
            tfrecord_writer.write(example.SerializeToString())
            
        except IOError  as e:
            print ('could not read:' , filenames[1])
            print ('error:' , e)
            print ('skip it \n')
            
        except cv2.error as e:
            print ('could not read:' , filenames[1])
            print ('error:' , e)
            print ('skip it \n')    
              
convert_dataset('train' , training_filenames)
convert_dataset('test' , testing_filenames)