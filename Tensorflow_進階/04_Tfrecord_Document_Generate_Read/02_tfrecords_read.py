#### 請解壓縮data.rar，取得本程式之數據 #####
import tensorflow as tf
import cv2  

tfrecord_name = 'cat_train.tfrecords'

# 生成一個queue隊列
filename_queue = tf.train.string_input_producer([tfrecord_name])

reader = tf.TFRecordReader()

# 返回文件名和文件
_ , serialized_example = reader.read(filename_queue)

# 將image數據和label取出來
features = tf.parse_single_example(serialized_example ,
                                   features = {'label': tf.FixedLenFeature([] , tf.int64),
                                               'image' : tf.FixedLenFeature([] , tf.string)})

# tf.decode_raw函數的意思是將原來編碼為字符串類型的變量重新變回來
img_t = tf.decode_raw(features['image'] , tf.uint8) 

# reshape為224*224的3通道image
img_t = tf.reshape(img_t , [224 , 224 , 3]) 

# 在flow中輸出label張量
label_t = tf.cast(features['label'] , tf.int32) 


# 以"批量讀取"的方式讀取tfrecord文件
# 利用num_threads個線程batch_size筆data
# 切記capacity一定要比min_after_dequeue大
img_batch , label_batch = tf.train.shuffle_batch([img_t , label_t] , 
                                                 batch_size = 5 ,
                                                 num_threads = 4,
                                                 capacity = 200 ,
                                                 min_after_dequeue = 10)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)
    for batch_i in range(0 , 10):
        image , label = sess.run([img_batch , label_batch]) #在sess中取出image和label
        print(label)
        
        # 將從cat_train.tfrecords讀出的image存檔，以供檢查
        for j in range(0 , 5):
            cv2.imwrite(str(batch_i * 5 + j) + '.jpg' , image[j , : , : , :])     
    
    coord.request_stop()
    coord.join(threads)
    
    
# 以普通的方式讀取tfrecord文件
#with tf.Session() as sess:
#    sess.run(tf.initialize_all_variables())
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord = coord)
#    for i in range(0 , 280):
#        image , label = sess.run([img_t , label_t]) #在會話中取出image和label
#        cv2.imwrite(str(i) + '.jpg' , image)     
#    coord.request_stop()
#    coord.join(threads)
    