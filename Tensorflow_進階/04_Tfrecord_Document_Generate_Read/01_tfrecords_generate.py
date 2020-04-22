#### 請解壓縮data.rar，取得本程式之數據 #####
import tensorflow as tf
import os
import random
import sys
import numpy as np
import cv2

DATASET_DIR = os.listdir()
tfrecord_name = 'cat_train.tfrecords'

# 若cat_train.tfrecords存在，則先刪除
if tfrecord_name in DATASET_DIR:
    os.remove(tfrecord_name) 
    
classes = ['tiger' , 'kittycat']
filename = []
for i in range(0 , len(classes)):
    dataset = os.listdir(os.path.join(classes[i]))
    for j in range(0 , len(dataset)):
        filename.append([classes[i] , dataset[j]]) # filename[k][0] -> 分類資料夾的名字
                                                   # filename[k][0] -> 分類資料夾中照片的檔名 
# 打亂順序
random.shuffle(filename)

# 生成cat_train.tfrecords文件
tfrecord_writer = tf.python_io.TFRecordWriter('cat_train.tfrecords') 


def bytes_feature(values):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [values]))

def int64_feature(values):
    # 檢查values是否為tuple或list
    if not isinstance(values , (tuple , list)) :
        values = [values]
    return tf.train.Feature(int64_list = tf.train.Int64List(value = values))

# image轉載成tfexample函數
def image_to_tfexample(image_data , label):
    return tf.train.Example(features = tf.train.Features(feature={'image' : bytes_feature(image_data) , 
                                                                  'label' : int64_feature(label) }))
    
for k in range(0 , len(filename)):   
    try:
        sys.stdout.write('\r>> Converting image' + str(k + 1) + '/' + str(len(filename)))
        sys.stdout.flush()
        
        #讀取image
        img_dir = os.path.join(filename[k][0] , filename[k][1])
        image_data = cv2.imread(img_dir)
        
        # 對image進行resize
        image_data = cv2.resize(image_data , (224 , 224) , interpolation = cv2.INTER_CUBIC)
       
        # 灰度化
        # image_data = cv2.cvtColor(image_data , cv2.COLOR_BGR2GRAY)
        
        # 將image轉為bytes
        image_data = image_data.tobytes()
        
        # 獲取label ('tiger' -> 0 , 'kittycat' -> 1)
        if filename[k][0] == 'tiger':
            label = 0
        elif filename[k][0] == 'kittycat':
            label = 1    
        
        # 將image_data和labele數據封裝在example
        example = image_to_tfexample(image_data , label)

        tfrecord_writer.write(example.SerializeToString())
 
    except IOError as e:
        print ('could not read:' , filename[k][1])
        print ('error:' , e)
        print ('skip it \n')
        
    except cv2.error as e:
        print ('could not read:' , filename[k][1])
        print ('error:' , e)
        print ('skip it \n')
      