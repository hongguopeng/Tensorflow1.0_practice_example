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

# 在trunk(主幹)先共享參數
class network_trunk(object):
    def __init__(self):
        self.xs = tf.placeholder(tf.float32 , [None , IMAGE_HEIGHT , IMAGE_WIDTH])
        self.xs_ = tf.expand_dims(self.xs , -1)
        self.on_train = tf.placeholder(tf.bool)
        self.lr = tf.placeholder(tf.float32)
        
        with tf.variable_scope('trunk'):
            self.network()
                 
       
    @staticmethod 
    def weight_variable(shape , name):
    #    initializer = tf.truncated_normal(shape , stddev = 0.0001)
    #    return tf.Variable(initializer)
        initializer = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(initializer = initializer(shape) , name = 'weight_{}'.format(name))
        
    @staticmethod 
    def bias_variable(shape , name):
        initializer = tf.zeros(shape) + 0.0001
        return tf.get_variable(initializer = initializer , name = 'bias_{}'.format(name))
    
    @staticmethod 
    def conv2d(x , W):
        # stride [1, x_movement , y_movement, 1]
        # Must have strides[0] = strides[3] = 1	
        return tf.nn.conv2d(x , W , strides = [1 , 1 , 1 , 1] , padding = 'SAME') 
    
    @staticmethod 
    def max_pool(x , k , s):
        # 不需要跟tf.nn.conv2d一樣要輸入W
        # ksize = [1 , *2* , *2* , 1] 輸入 2 , 2 代表每2x2個pixel做一次選取pixel最大的動作
        # stride [1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x , ksize = [1 , k , k , 1] , strides = [1 , s , s , 1] , padding = 'SAME')
    
    @staticmethod 
    def batch_norm_layer(inputs , on_train , convolution , name):
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
        scale = tf.get_variable(initializer = tf.ones([1 , inputs.shape[-1].value]) , name = 'scale_{}'.format(name))
        shift = tf.get_variable(initializer = tf.zeros([1 , inputs.shape[-1].value]) , name = 'shift_{}'.format(name))
        temp = (inputs - mean) / tf.sqrt(var + 0.001)
        outputs = tf.multiply(temp , scale) + shift
      
        return outputs , ema_apply_op 
        
    def network(self) :
        # conv1 layer 
        conv_w_1 = self.weight_variable([3 , 3 , 1 , 32] , 'layer_1') 
        conv_b_1 = self.bias_variable([32] , 'layer_1')
        conv_output_1 = tf.nn.relu(self.conv2d(self.xs_ , conv_w_1) + conv_b_1) 
        conv_bn_1 , conv_ema_1 = self.batch_norm_layer(conv_output_1 , self.on_train , True , 'layer_1')
        conv_pooling_1 = self.max_pool(conv_bn_1 , k = 3 , s = 3)
           
        # conv2 layer #
        conv_w_2 = self.weight_variable([3 , 3 , 32 , 64]  , 'layer_2') 
        conv_b_2 = self.bias_variable([64]  , 'layer_2')
        conv_output_2 = tf.nn.relu(self.conv2d(conv_pooling_1 , conv_w_2) + conv_b_2)
        conv_bn_2 , conv_ema_2 = self.batch_norm_layer(conv_output_2 , self.on_train , True  , 'layer_2')
        
        # conv3 layer 
        conv_w_3 = self.weight_variable([3 , 3 , 64 , 64] , 'layer_3') 
        conv_b_3 = self.bias_variable([64] , 'layer_3')
        conv_output_3 = tf.nn.relu(self.conv2d(conv_bn_2 , conv_w_3) + conv_b_3) 
        conv_bn_3 , conv_ema_3 = self.batch_norm_layer(conv_output_3 , self.on_train , True , 'layer_3')
        conv_pooling_3 = self.max_pool(conv_bn_3 , k = 2 , s = 2)
          
        # conv4 layer 
        conv_w_4 = self.weight_variable([3 , 3 , 64 , 128]  , 'layer_4') 
        conv_b_4 = self.bias_variable([128]  , 'layer_4')
        conv_output_4 = tf.nn.relu(self.conv2d(conv_pooling_3 , conv_w_4) + conv_b_4) 
        conv_bn_4 , conv_ema_4 = self.batch_norm_layer(conv_output_4 , self.on_train , True  , 'layer_4')          
        
        # conv5 layer 
        conv_w_5 = self.weight_variable([3 , 3 , 128 , 64]  , 'layer_5') 
        conv_b_5 = self.bias_variable([64]  , 'layer_5')
        conv_output_5 = tf.nn.relu(self.conv2d(conv_bn_4 , conv_w_5) + conv_b_5) 
        conv_bn_5 , conv_ema_5 = self.batch_norm_layer(conv_output_5 , self.on_train , True  , 'layer_5')
        conv_pooling_5 = self.max_pool(conv_bn_5 , k = 2 , s = 2)
        self.conv_pooling_5_flatten = tf.layers.flatten(conv_pooling_5)
        self.ema = [conv_ema_1 , conv_ema_2 , conv_ema_3 , conv_ema_4 , conv_ema_5]
 
# 在branch(支幹)不共享參數，分別針對驗證碼的4個數字做訓練  
class network_branch(object): 
    def __init__(self , branch , trunk_end):
        self.on_train = tf.placeholder(tf.bool)
        self.y = tf.placeholder(tf.int32 , [None , 4])
        self.y_one_hot = tf.one_hot(self.y[: , branch] , depth = CHAR_SET_LEN)
        self.conv_pooling_5_flatten = trunk_end
        
        with tf.variable_scope('branch_{}'.format(branch)):
            self.network()
        
    @staticmethod 
    def weight_variable(shape , name):
    #    initializer = tf.truncated_normal(shape , stddev = 0.0001)
    #    return tf.Variable(initializer)
        initializer = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(initializer = initializer(shape) , name = 'weight_{}'.format(name))
        
    @staticmethod 
    def bias_variable(shape , name):
        initializer = tf.zeros(shape) + 0.0001
        return tf.get_variable(initializer = initializer , name = 'bias_{}'.format(name))
    
    @staticmethod 
    def conv2d(x , W):
        # stride [1, x_movement , y_movement, 1]
        # Must have strides[0] = strides[3] = 1	
        return tf.nn.conv2d(x , W , strides = [1 , 1 , 1 , 1] , padding = 'SAME') 
    
    @staticmethod 
    def max_pool(x , k , s):
        # 不需要跟tf.nn.conv2d一樣要輸入W
        # ksize = [1 , *2* , *2* , 1] 輸入 2 , 2 代表每2x2個pixel做一次選取pixel最大的動作
        # stride [1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x , ksize = [1 , k , k , 1] , strides = [1 , s , s , 1] , padding = 'SAME')
    
    @staticmethod 
    def batch_norm_layer(inputs , on_train , convolution , name):
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
        scale = tf.get_variable(initializer = tf.ones([1 , inputs.shape[-1].value]) , name = 'scale_{}'.format(name))
        shift = tf.get_variable(initializer = tf.zeros([1 , inputs.shape[-1].value]) , name = 'shift_{}'.format(name))
        temp = (inputs - mean) / tf.sqrt(var + 0.001)
        outputs = tf.multiply(temp , scale) + shift
          
        return outputs , ema_apply_op 
       
    def network(self):    
        fc_w = self.weight_variable([self.conv_pooling_5_flatten.shape[1].value , 10] , 'layer_1')
        fc_b = self.bias_variable([10] , 'layer_1')
        fc_output = tf.nn.relu(tf.matmul(self.conv_pooling_5_flatten , fc_w) + fc_b)
        fc_bn , fc_ema = self.batch_norm_layer(fc_output , self.on_train , False , 'layer_1')
        self.ema = fc_ema
        fc_dropout = tf.cond(self.on_train , 
                             lambda : tf.nn.dropout(fc_bn , keep_prob = 0.8) , 
                             lambda : tf.nn.dropout(fc_bn , keep_prob = 1))
        prediction = tf.nn.softmax(fc_dropout)        
        cross_entropy = self.y_one_hot * tf.log(prediction + 1e-9)  
        self.cross_entropy = -tf.reduce_mean(tf.reduce_sum(cross_entropy , axis = 1)) 
        correct = tf.equal(tf.argmax(self.y_one_hot , 1) , tf.argmax(prediction , 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct , tf.float32))

trunk = network_trunk()
brahch_list = [network_branch(i , trunk.conv_pooling_5_flatten) for i in range(0 , 4)]

# 收集4個branch的Optimizer
train_op = [tf.train.AdamOptimizer(trunk.lr).minimize(brahch_list[i].cross_entropy) for i in range(0 , 4)]
train_op = tf.group(train_op)

# 收集trunk與4個branch的ema
update_ema = tf.group(trunk.ema , brahch_list[0].ema , brahch_list[1].ema , brahch_list[2].ema , brahch_list[3].ema) 

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
lr = 0.0005
for epoch_i in range(0 , 400):
    shuffle = np.arange(2700)
    np.random.shuffle(shuffle)
    trainX = trainX[shuffle]
    trainY = trainY[shuffle]
    
    for batch_i in range(0 , 54):
        sess.run([train_op , update_ema] ,
                 feed_dict = {trunk.xs : trainX[index[batch_i]] , 
                              brahch_list[0].y : trainY[index[batch_i]] , 
                              brahch_list[1].y : trainY[index[batch_i]] ,
                              brahch_list[2].y : trainY[index[batch_i]] ,
                              brahch_list[3].y : trainY[index[batch_i]] ,
                              trunk.on_train : True , 
                              brahch_list[0].on_train : True ,
                              brahch_list[1].on_train : True ,
                              brahch_list[2].on_train : True ,
                              brahch_list[3].on_train : True , 
                              trunk.lr : lr})

        # 每經過5個batch，計算一次loss和準確率
        if batch_i % 5 == 0:
            loss0 , loss1 , loss2 , loss3 = \
            sess.run([brahch_list[0].cross_entropy , brahch_list[1].cross_entropy , brahch_list[2].cross_entropy , brahch_list[3].cross_entropy] , 
                     feed_dict = {trunk.xs : trainX[index[batch_i]] , 
                                  brahch_list[0].y : trainY[index[batch_i]] , 
                                  brahch_list[1].y : trainY[index[batch_i]] ,
                                  brahch_list[2].y : trainY[index[batch_i]] ,
                                  brahch_list[3].y : trainY[index[batch_i]] ,
                                  trunk.on_train : False , 
                                  brahch_list[0].on_train : False ,
                                  brahch_list[1].on_train : False ,
                                  brahch_list[2].on_train : False ,
                                  brahch_list[3].on_train : False})  
            acc0 , acc1 , acc2 , acc3 = \
            sess.run([brahch_list[0].accuracy , brahch_list[1].accuracy , brahch_list[2].accuracy , brahch_list[3].accuracy] , 
                     feed_dict = {trunk.xs : trainX[index[batch_i]] , 
                                  brahch_list[0].y : trainY[index[batch_i]] , 
                                  brahch_list[1].y : trainY[index[batch_i]] ,
                                  brahch_list[2].y : trainY[index[batch_i]] ,
                                  brahch_list[3].y : trainY[index[batch_i]] ,
                                  trunk.on_train : False , 
                                  brahch_list[0].on_train : False ,
                                  brahch_list[1].on_train : False ,
                                  brahch_list[2].on_train : False ,
                                  brahch_list[3].on_train : False}) 
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
            lr = lr * 0.9

    loss0_test , loss1_test , loss2_test , loss3_test = \
    sess.run([brahch_list[0].cross_entropy , brahch_list[1].cross_entropy , brahch_list[2].cross_entropy , brahch_list[3].cross_entropy] , 
             feed_dict = {trunk.xs : testX , 
                          brahch_list[0].y : testY , 
                          brahch_list[1].y : testY ,
                          brahch_list[2].y : testY ,
                          brahch_list[3].y : testY ,
                          trunk.on_train : False , 
                          brahch_list[0].on_train : False ,
                          brahch_list[1].on_train : False ,
                          brahch_list[2].on_train : False ,
                          brahch_list[3].on_train : False})       
    acc0_test , acc1_test , acc2_test , acc3_test = \
    sess.run([brahch_list[0].accuracy , brahch_list[1].accuracy , brahch_list[2].accuracy , brahch_list[3].accuracy] , 
             feed_dict = {trunk.xs : testX , 
                          brahch_list[0].y : testY , 
                          brahch_list[1].y : testY ,
                          brahch_list[2].y : testY ,
                          brahch_list[3].y : testY ,
                          trunk.on_train : False , 
                          brahch_list[0].on_train : False ,
                          brahch_list[1].on_train : False ,
                          brahch_list[2].on_train : False ,
                          brahch_list[3].on_train : False})          
    
    print('*' * 30)
    print('epoch_i : {}'.format(epoch_i))
    print('testing_loss : {:.2f} , {:.2f} , {:.2f} , {:.2f}'\
          .format(loss0_test , loss1_test , loss2_test , loss3_test))
    print('testing_accuracy : {:.2%} , {:.2%} , {:.2%} , {:.2%}'\
          .format(acc0_test , acc1_test , acc2_test , acc3_test) )
    print('*' * 30 , '\n')
    
    test_loss_acc['loss'].append([loss0_test , loss1_test , loss2_test , loss3_test])
    test_loss_acc['acc'].append([acc0_test , acc1_test , acc2_test , acc3_test])
