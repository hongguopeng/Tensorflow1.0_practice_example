#### 請解壓縮data.rar，取得本程式之數據 #####
# 驗證碼識別(訓練)
import os
import tensorflow as tf
import numpy as np
import random
import cv2
import math
from sklearn.model_selection import train_test_split


# 不同字符數量
CHAR_SET_LEN = 10
# 批次
BATCH_SIZE = 50
# 圖片高度
IMAGE_HEIGHT = 60
# 圖片寬度
IMAGE_WIDTH = 160


# 獲取數據路徑
imagePaths = [] 
for files in os.listdir('./captcha/images'):
    imagePaths.append('./captcha/images/{}'.format(files))
random.seed(42)
random.shuffle(imagePaths)

data = []
label = [[] for _ in range(0 , 4)]
# 獲取數據標簽
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_WIDTH , IMAGE_HEIGHT))
    image = cv2.cvtColor( image , cv2.COLOR_BGR2GRAY)
    data.append(image)
    label_temp = imagePath.split('/')[-1].split('.')[0]
    for i in range(0 , 4):
        label[i].append(int(label_temp[i]))
data = np.array(data).astype(np.float32)        
label = np.array(label).T

# 數據集切分
(trainX , testX , trainY , testY) = train_test_split(data,
                                                     label , 
                                                     test_size = 0.1, 
                                                     random_state = 42)

def weight_variable(shape):
#    initializer = tf.truncated_normal(shape , stddev = 0.0001)
#    return tf.Variable(initializer)
    initializer = tf.contrib.layers.xavier_initializer() # 效果比tf.truncated_normal還要好
    return tf.Variable(initializer(shape))
    
def bias_variable(shape):
    initial = tf.constant(0.0001 , shape = shape) 
    return tf.Variable(initial)

def conv2d(x , W):
    # stride [1, x_movement , y_movement, 1]
    # Must have strides[0] = strides[3] = 1	
    return tf.nn.conv2d(x , W , strides = [1 , 1 , 1 , 1] , padding = 'SAME') 

def max_pool(x , k , s):
    # 不需要跟tf.nn.conv2d一樣要輸入W
    # ksize = [1 , *2* , *2* , 1] 輸入 2 , 2 代表每2x2個pixel做一次選取pixel最大的動作
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x , ksize = [1 , k , k , 1] , strides = [1 , s , s , 1] , padding = 'SAME')

def batch_norm_layer(inputs , on_train , convolution):
    # the dimension you wanna normalize, here [0] for batch
    # for image, you wanna do [0 , 1 , 2] for [batch , height , width] but not channel
    if convolution:
        fc_mean , fc_var = tf.nn.moments(inputs , axes = [0 , 1 , 2] , name = 'mean_var')
    else:
        fc_mean , fc_var = tf.nn.moments(inputs , axes = [0] , name = 'mean_var')
    
    ema = tf.train.ExponentialMovingAverage(decay = 0.75)
    ema_apply_op = ema.apply([fc_mean , fc_var])
    mean = tf.cond(on_train , lambda : fc_mean , lambda : ema.average(fc_mean))
    var = tf.cond(on_train , lambda : fc_var , lambda : ema.average(fc_var))
    
    scale = tf.Variable(tf.ones([1 , inputs.shape[-1].value]) , name = 'scale') 
    shift = tf.Variable(tf.zeros([1 , inputs.shape[-1].value]) , name = 'shift')
    temp = (inputs - mean) / tf.sqrt(var + 0.001)
    outputs = tf.multiply(temp , scale) + shift
  
    return outputs , ema_apply_op 

xs = tf.placeholder(tf.float32 , [None , IMAGE_HEIGHT , IMAGE_WIDTH])
xs_ = tf.expand_dims(xs , -1)
y = tf.placeholder(tf.int32 , [None , 4])
lr = tf.placeholder(tf.float32)
y_one_hot_0 = tf.one_hot(y[: , 0] , depth = CHAR_SET_LEN)
y_one_hot_1 = tf.one_hot(y[: , 1] , depth = CHAR_SET_LEN)
y_one_hot_2 = tf.one_hot(y[: , 2] , depth = CHAR_SET_LEN)
y_one_hot_3 = tf.one_hot(y[: , 3] , depth = CHAR_SET_LEN)
on_train = tf.placeholder(tf.bool) # train/test selector for batch normalisation

# 在trunk(主幹)先共享參數
with tf.variable_scope('trunk'):
    # conv1 layer 
    conv_w_1 = weight_variable([3 , 3 , 1 , 32]) 
    conv_b_1 = bias_variable([32])
    conv_output_1 = tf.nn.relu(conv2d(xs_ , conv_w_1) + conv_b_1) 
    conv_bn_1 , conv_ema_1 = batch_norm_layer(conv_output_1 , on_train , True)
    conv_pooling_1 = max_pool(conv_bn_1 , k = 3 , s = 3)
       
    # conv2 layer #
    conv_w_2 = weight_variable([3 , 3 , 32 , 64]) 
    conv_b_2 = bias_variable([64])
    conv_output_2 = tf.nn.relu(conv2d(conv_pooling_1 , conv_w_2) + conv_b_2)
    conv_bn_2 , conv_ema_2 = batch_norm_layer(conv_output_2 , on_train , True)
    
    # conv3 layer 
    conv_w_3 = weight_variable([3 , 3 , 64 , 64]) 
    conv_b_3 = bias_variable([64])
    conv_output_3 = tf.nn.relu(conv2d(conv_bn_2 , conv_w_3) + conv_b_3) 
    conv_bn_3 , conv_ema_3 = batch_norm_layer(conv_output_3 , on_train , True)
    conv_pooling_3 = max_pool(conv_bn_3 , k = 2 , s = 2)
      
    # conv4 layer 
    conv_w_4 = weight_variable([3 , 3 , 64 , 128]) 
    conv_b_4 = bias_variable([128])
    conv_output_4 = tf.nn.relu(conv2d(conv_pooling_3 , conv_w_4) + conv_b_4) 
    conv_bn_4 , conv_ema_4 = batch_norm_layer(conv_output_4 , on_train , True)
    
    # conv5 layer 
    conv_w_5 = weight_variable([3 , 3 , 128 , 64]) 
    conv_b_5 = bias_variable([64])
    conv_output_5 = tf.nn.relu(conv2d(conv_bn_4 , conv_w_5) + conv_b_5) 
    conv_bn_5 , conv_ema_5 = batch_norm_layer(conv_output_5 , on_train , True)
    conv_pooling_5 = max_pool(conv_bn_5 , k = 2 , s = 2)
    conv_pooling_5_flatten = tf.layers.flatten(conv_pooling_5)

# 在branch(支幹)不共享參數，分別針對驗證碼的4個數字做訓練
with tf.variable_scope('branch'):
    with tf.variable_scope('branch_0'):
        fc_w_0 = weight_variable([conv_pooling_5_flatten.shape[1].value , 10])
        fc_b_0 = bias_variable([10])
        fc_output_0 = tf.nn.relu(tf.matmul(conv_pooling_5_flatten , fc_w_0) + fc_b_0)
        fc_bn_0 , fc_ema_0 = batch_norm_layer(fc_output_0 , on_train , False)
        fc_dropout_0 = tf.cond(on_train , 
                               lambda : tf.nn.dropout(fc_bn_0 , keep_prob = 0.8) , 
                               lambda : tf.nn.dropout(fc_bn_0 , keep_prob = 1))
        prediction0 = tf.nn.softmax(fc_dropout_0)
        cross_entropy_0 = y_one_hot_0 * tf.log(prediction0 + 1e-9)  
        cross_entropy_0 = -tf.reduce_mean(tf.reduce_sum(cross_entropy_0 , axis = 1))
    
    
    with tf.variable_scope('branch_1'):
        fc_w_1 = weight_variable([conv_pooling_5_flatten.shape[1].value , 10])
        fc_b_1 = bias_variable([10])
        fc_output_1 = tf.nn.relu(tf.matmul(conv_pooling_5_flatten , fc_w_1) + fc_b_1)
        fc_bn_1 , fc_ema_1 = batch_norm_layer(fc_output_1 , on_train , False)
        fc_dropout_1 = tf.cond(on_train , 
                               lambda : tf.nn.dropout(fc_bn_1 , keep_prob = 0.95) , 
                               lambda : tf.nn.dropout(fc_bn_1 , keep_prob = 1))
        prediction1 = tf.nn.softmax(fc_dropout_1)
        cross_entropy_1 = y_one_hot_1 * tf.log(prediction1 + 1e-9)  
        cross_entropy_1 = -tf.reduce_mean(tf.reduce_sum(cross_entropy_1 , axis = 1))
        
    
    with tf.variable_scope('branch_2'):
        fc_w_2 = weight_variable([conv_pooling_5_flatten.shape[1].value , 10])
        fc_b_2 = bias_variable([10])
        fc_output_2 = tf.nn.relu(tf.matmul(conv_pooling_5_flatten , fc_w_2) + fc_b_2)
        fc_bn_2 , fc_ema_2 = batch_norm_layer(fc_output_2 , on_train , False)
        fc_dropout_2 = tf.cond(on_train , 
                               lambda : tf.nn.dropout(fc_bn_2 , keep_prob = 0.95) , 
                               lambda : tf.nn.dropout(fc_bn_2 , keep_prob = 1))
        prediction2 = tf.nn.softmax(fc_dropout_2)
        cross_entropy_2 = y_one_hot_2 * tf.log(prediction2 + 1e-9)  
        cross_entropy_2 = -tf.reduce_mean(tf.reduce_sum(cross_entropy_2 , axis = 1))
    
    
    with tf.variable_scope('branch_3'):
        fc_w_3 = weight_variable([conv_pooling_5_flatten.shape[1].value , 10])
        fc_b_3 = bias_variable([10])
        fc_output_3 = tf.nn.relu(tf.matmul(conv_pooling_5_flatten , fc_w_3) + fc_b_3)
        fc_bn_3 , fc_ema_3 = batch_norm_layer(fc_output_3 , on_train , False)
        fc_dropout_3 = tf.cond(on_train , 
                               lambda : tf.nn.dropout(fc_bn_3 , keep_prob = 0.95) , 
                               lambda : tf.nn.dropout(fc_bn_3 , keep_prob = 1))
        prediction3 = tf.nn.softmax(fc_dropout_3)
        cross_entropy_3 = y_one_hot_3 * tf.log(prediction3 + 1e-9)  
        cross_entropy_3 = -tf.reduce_mean(tf.reduce_sum(cross_entropy_3 , axis = 1))

# 收集trunk與4個branch的ema
ema_list = [conv_ema_1 , conv_ema_2 , conv_ema_3 , conv_ema_4 , conv_ema_5 , fc_ema_0 , fc_ema_1 , fc_ema_2 , fc_ema_3]
update_ema = tf.group(ema_list)

# 收集4個branch的Optimizer
train_op_branch_0 = tf.train.AdamOptimizer(lr).minimize(cross_entropy_0)
train_op_branch_1 = tf.train.AdamOptimizer(lr).minimize(cross_entropy_1)
train_op_branch_2 = tf.train.AdamOptimizer(lr).minimize(cross_entropy_2)
train_op_branch_3 = tf.train.AdamOptimizer(lr).minimize(cross_entropy_3)
train_op = tf.group([train_op_branch_0 , train_op_branch_1 , train_op_branch_2 , train_op_branch_3])

# 分別針對驗證碼的4個數字計算準確率
correct0 = tf.equal(tf.argmax(y_one_hot_0 , 1) , tf.argmax(prediction0 , 1))
accuracy0 = tf.reduce_mean(tf.cast(correct0 , tf.float32))

correct1 = tf.equal(tf.argmax(y_one_hot_1 , 1) , tf.argmax(prediction1 , 1))
accuracy1 = tf.reduce_mean(tf.cast(correct1 , tf.float32))

correct2 = tf.equal(tf.argmax(y_one_hot_2 , 1) , tf.argmax(prediction2 , 1))
accuracy2 = tf.reduce_mean(tf.cast(correct2 , tf.float32))

correct3 = tf.equal(tf.argmax(y_one_hot_3 , 1) , tf.argmax(prediction3 , 1))
accuracy3 = tf.reduce_mean(tf.cast(correct3 , tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# minibatch data index
epochs = 50
batch_num = 100
step = (math.ceil(len(trainX) / BATCH_SIZE)) * BATCH_SIZE
temp = []
j = 0
index = []
for ii in range(0 , step):
    j = j + 1
    if j > len(trainX):
        j = j - len(trainX)   
    temp.append(j)  
    if len(temp) == BATCH_SIZE:
       index.append(temp)
       temp = []
index = list(np.array(index) - 1)

train_loss_acc = {'loss' : [] , 'acc' : []}
test_loss_acc = {'loss' : [] , 'acc' : []}
learning_rate = 0.0005
for epoch_i in range(0 , 400):
    shuffle = np.arange(2700)
    np.random.shuffle(shuffle)
    trainX = trainX[shuffle]
    trainY = trainY[shuffle]
    
    for batch_i in range(0 , 54):
        sess.run([train_op , update_ema] ,
                 feed_dict = {xs : trainX[index[batch_i]] , 
                              y : trainY[index[batch_i]] , 
                              on_train : True , 
                              lr : learning_rate})

        # 每經過5個batch，計算一次loss和準確率
        if batch_i % 5 == 0:
            loss0 , loss1 , loss2 , loss3 = \
            sess.run([cross_entropy_0 , cross_entropy_1 , cross_entropy_2 , cross_entropy_3] , 
                     feed_dict = {xs : trainX[index[batch_i]] , 
                                  y : trainY[index[batch_i]] , 
                                  on_train : False , 
                                  lr : learning_rate})  
            acc0 , acc1 , acc2 , acc3 = \
            sess.run([accuracy0 , accuracy1 , accuracy2 , accuracy3] , 
                     feed_dict = {xs : trainX[index[batch_i]] , 
                                  y : trainY[index[batch_i]] , 
                                  on_train : False , 
                                  lr : learning_rate}) 
            print('-'*30)
            print('epoch_i : {}'.format(epoch_i))
            print('batch_i : {}'.format(batch_i))
            print('training_loss : {:.2f} , {:.2f} , {:.2f} , {:.2f}'\
                  .format(loss0 , loss1 , loss2 , loss3))
            print('training_accuracy : {:.2%} , {:.2%} , {:.2%} , {:.2%}\n'\
                  .format(acc0 , acc1 , acc2 , acc3) )

            train_loss_acc['loss'].append([loss0 , loss1 , loss2 , loss3])
            train_loss_acc['acc'].append([acc0 , acc1 , acc2 , acc3])
        
        if (epoch_i * 54 + batch_i) % 1000 == 0 and (epoch_i * 54 + batch_i) > 0 : 
            learning_rate = learning_rate * 0.9

    loss0_test , loss1_test , loss2_test , loss3_test = \
    sess.run([cross_entropy_0 , cross_entropy_1 , cross_entropy_2 , cross_entropy_3] , 
             feed_dict = {xs : testX , y : testY , on_train : False})        
    acc0_test , acc1_test , acc2_test , acc3_test = \
    sess.run([accuracy0 , accuracy1 , accuracy2 , accuracy3] , 
             feed_dict = {xs : testX , y : testY , on_train : False})         
    
    print('*' * 30)
    print('epoch_i : {}'.format(epoch_i))
    print('testing_loss : {:.2f} , {:.2f} , {:.2f} , {:.2f}'\
          .format(loss0_test , loss1_test , loss2_test , loss3_test))
    print('testing_accuracy : {:.2%} , {:.2%} , {:.2%} , {:.2%}'\
          .format(acc0_test , acc1_test , acc2_test , acc3_test) )
    print('*' * 30 , '\n')
    
    test_loss_acc['loss'].append([loss0_test , loss1_test , loss2_test , loss3_test])
    test_loss_acc['acc'].append([acc0_test , acc1_test , acc2_test , acc3_test])
