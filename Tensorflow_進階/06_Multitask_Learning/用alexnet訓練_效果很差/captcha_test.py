# 驗證碼測試
import os
import tensorflow as tf
from PIL import Image
from nets import nets_factory
import numpy as np
import matplotlib.pyplot as plt

# 不同字符數量
CHAR_SET_LEN = 10
# 圖片高度
IMAGE_HEIGHT = 60
# 圖片寬度
IMAGE_WIDTH = 160
# 批次
BATCH_SIZE = 1
# tfrecord文件存放路徑
TFRECORD_FILE = 'E:/SVN/Gavin/Learn/Python/pygame/captcha/test.tfrecords'

# placeholder
x = tf.placeholder(tf.float32,[None,224,224])

# 從tfrecord讀出數據
def read_and_decode(filename):
    # 根據文件名生成一個隊列
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader從文件隊列中讀入一個序列化的樣本,返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符號化的樣本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label0': tf.FixedLenFeature([], tf.int64),
            'label1': tf.FixedLenFeature([], tf.int64),
            'label2': tf.FixedLenFeature([], tf.int64),
            'label3': tf.FixedLenFeature([], tf.int64),
        }
    )
    img = features['image']
    # 獲取圖片數據
    image = tf.decode_raw(img, tf.uint8)
    # 沒有經過預處理的灰度圖
    image_raw = tf.reshape(image, [224,224])
    # 圖片預處理
    image = tf.cast(image, tf.float32) /255.0
    image = tf.subtract(image,0.5)
    image = tf.multiply(image,2.0)
    # 獲取label
    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)
    return image, image_raw,label0,label1,label2,label3


# 獲取圖片數據和標簽
image, image_raw,label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)
print(image,image_raw,label0,label1,label2, label3)
# 使用shuffle_batch可以隨機打亂輸入 next_batch挨著往下取
# shuffle_batch才能實現[img,label]的同步,也即特征和label的同步,不然可能輸入的特征和label不匹配
# 比如只有這樣使用,才能使img和label一一對應,每次提取一個image和對應的label
# shuffle_batch返回的值就是RandomShuffleQueue.dequeue_many()的結果
# Shuffle_batch構建了一個RandomShuffleQueue，並不斷地把單個的[img,label],送入隊列中
img_batch,img_raw_batch, label_batch0,label_batch1,label_batch2,label_batch3 = tf.train.shuffle_batch(
                                         [image,image_raw, label0,label1,label2,label3],
                                        batch_size=BATCH_SIZE, capacity=5000,
                                        min_after_dequeue=1000,num_threads=1)
# 定義網絡結構
train_network_fn = nets_factory.get_network_fn(
    'alexnet_v2',
    num_classes=CHAR_SET_LEN,
    weight_decay=0.0005,
    is_training=False
)

with tf.Session() as sess:
    X = tf.reshape(x,[BATCH_SIZE,224,224,1])
    # 數據輸入網絡得到輸出值
    logits0,logits1,logits2,logits3,end_points = train_network_fn(X)

    # 預測值
    predict0 = tf.reshape(logits0,[-1,CHAR_SET_LEN])
    predict0 = tf.argmax(predict0,1)

    predict1 = tf.reshape(logits1, [-1, CHAR_SET_LEN])
    predict1 = tf.argmax(predict1, 1)

    predict2 = tf.reshape(logits2, [-1, CHAR_SET_LEN])
    predict2 = tf.argmax(predict2, 1)

    predict3 = tf.reshape(logits3, [-1, CHAR_SET_LEN])
    predict3 = tf.argmax(predict3, 1)


    # 初始化
    sess.run(tf.global_variables_initializer())
    # 創建一個協調器，管理線程
    coord = tf.train.Coordinator()
    # 啟動隊列
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    for i in range(10):
        # 獲取一個批次的數據和標簽
        b_image,b_image_raw, b_label0,b_label1,b_label2,b_label3 = sess.run([img_batch,img_raw_batch,
                                                                 label_batch0, label_batch1, label_batch2, label_batch3])
        # 顯示圖片
        img = Image.fromarray(b_image_raw[0],'L')
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        # 打印標簽
        print('label:',b_label0,b_label1,b_label2,b_label3)
        # 預測
        label0,label1,label2,label3 = sess.run([predict0,predict1,predict2,predict3],
                                               feed_dict={x:b_image})
        # print
        print('predict:',label0,label1,label2,label3)

        # 通知其他線程關閉
        coord.request_stop()
        # 其他所有線程關閉之後，這一函數才能返回
        coord.join(threads)
