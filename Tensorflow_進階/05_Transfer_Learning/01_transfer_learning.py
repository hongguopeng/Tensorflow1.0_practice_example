#### 請解壓縮data.rar，取得本程式之數據 #####
import os
import numpy as np
import tensorflow as tf
import math
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt

# 只是下載tiger與kittycat的image，並不是太重要
#from urllib.request import urlretrieve
#def download():     
#    categories = ['tiger', 'kittycat']
#    for ii in range(0 , 2):    
#        if not os.path.isdir(categories[ii]):
#            os.mkdir(categories[ii])
#        
#        path = os.path.join('imagenet_' + categories[ii] + '.txt')         
#        file = open(path , 'r')
#        urls = file.readlines()
#        for i in range(0 , len(urls)):  
#            text = urls[i].replace('\n' , '')
#            try:
#                urlretrieve(text , os.path.join(categories[ii] , text.split('/')[-1]))
#                print(categories[ii] + ' ' + str(i) + '/' + str(len(urls))) 
#                img = skimage.io.imread(os.path.join(categories[ii] , text.split('/')[-1]))
#                img = img / 255.0
#                if img[1 : -1 , 1 : -1].astype('int').sum() == 182771:
#                    os.remove(os.path.join(categories[ii] , text.split('/')[-1]))
#                    print(os.path.join(categories[ii] , text.split('/')[-1]) + ' ' + 'is' + ' ' + 'removed')
#            except:
#                print(categories[ii] + ' ' + str(i) + '/' + str(len(urls)) + ' ' + 'no image')


def load_img(path):
    try:
        img = skimage.io.imread(path)
        img = img / 255.0 # 對image做normalization
        
        # 對中心截取正方形的image
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        truncate_img = img[yy : yy + short_edge , xx : xx + short_edge]
        
        # 對truncate_img做resize成(224 , 224 , 3)
        resized_img = skimage.transform.resize(truncate_img , [224 , 224 , 3])
        
        # 再讓resized_img多一個維度
        resized_img = np.expand_dims(resized_img , axis = 0)
   
        return resized_img
    except OSError:
        pass

# 分別將做完處理的image先存到imgs_dict中        
imgs = ['tiger' , 'kittycat']
imgs_dict = {'tiger' : [] , 'kittycat' : []}
for class_ in imgs_dict.keys():
    count = 0
    for file in os.listdir(class_):
        resized_img = load_img(os.path.join(class_ , file))
        if resized_img is not None:
            imgs_dict[class_].append(resized_img) # [1 , height , width , depth] * n 
        if len(imgs_dict[class_]) == 800:         # 只收集400張image做訓練
            break

# tigers_x => (400 , 224 , 224 , 3)
tigers_x = np.zeros([len(imgs_dict['tiger']) , 224 , 224 , 3])        
for t in range(0 , len(imgs_dict['tiger'])):
    tigers_x[t , : , : , :] = imgs_dict['tiger'][t]

# cats_x => (400 , 224 , 224 , 3)    
cats_x = np.zeros([len(imgs_dict['kittycat']) , 224 , 224 , 3])        
for t in range(0 , len(imgs_dict['kittycat'])):
    cats_x[t , : , : , :] = imgs_dict['kittycat'][t]   

# 對tiger與kittycat的身體長度做個隨機取樣
# 但兩者取值的範圍不一樣，tiger取樣的結果一定會比kittycat還大   
# 而本程式就是要輸入tiger與kittycat的image，並預測tiger與kittycat身體的長度
tigers_y = np.maximum(20 , np.random.randn(len(imgs_dict['tiger']) , 1) * 30 + 100)
cats_y = np.maximum(10 , np.random.randn(len(imgs_dict['kittycat']) , 1) * 8 + 40)

xs = np.concatenate([tigers_x , cats_x] , axis = 0)
ys = np.concatenate([tigers_y , cats_y] , axis = 0)



# 讀取vgg16.npy
vgg16_npy_path = os.path.join('vgg16.npy')
# data_dict包含['conv5_1', 'fc6', 'conv5_3', 'conv5_2', 'fc8', 'fc7', 'conv4_1', 'conv4_2', 'conv4_3', 'conv3_3', 'conv3_2', 'conv3_1', 'conv1_1', 'conv1_2', 'conv2_2', 'conv2_1']
# 以'conv5_1'為例子，data_dict['conv5_1']包含一個weight與bias
data_dict = np.load(vgg16_npy_path , encoding = 'latin1').item()

def conv_layer(bottom , name):
    with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
        weight = tf.Variable(data_dict[name][0] , name = 'weight')
        bias = tf.Variable(data_dict[name][1] , name = 'bias')
        conv = tf.nn.conv2d(bottom , weight , [1 , 1 , 1 , 1] , padding = 'SAME')
        lout = tf.nn.relu(conv + bias)
        return lout
    
def max_pool(bottom , name):
    return tf.nn.max_pool(bottom , ksize = [1 , 2 , 2 , 1] , strides = [1 , 2 , 2 , 1] , padding = 'SAME', name = name)


with tf.variable_scope('input_layer'):
    input_x = tf.placeholder(tf.float32 , [None , 224 , 224 , 3] , name = 'xs')
    output_y = tf.placeholder(tf.float32 , [None , 1] , name = 'ys')

    # 針對第3個維度拆開成3個tensor => red , green , blue
    # 每個tensor減去各自對應的值再合併成bgr
    vgg_mean = [103.939 , 116.779 , 123.68]
    red , green , blue = tf.split(axis = 3 , num_or_size_splits = 3 , value = input_x * 255.0)
    bgr = tf.concat(axis = 3 , values = [blue - vgg_mean[0] , green - vgg_mean[1] , red - vgg_mean[2]])


with tf.variable_scope('conv_layer'):
    conv1_1 = conv_layer(bgr , 'conv1_1')
    conv1_2 = conv_layer(conv1_1 , 'conv1_2')
    pool1 = max_pool(conv1_2 , 'pool1')
    
    conv2_1 = conv_layer(pool1 , 'conv2_1')
    conv2_2 = conv_layer(conv2_1 , 'conv2_2')
    pool2 = max_pool(conv2_2 , 'pool2')
    
    conv3_1 = conv_layer(pool2 , 'conv3_1')
    conv3_2 = conv_layer(conv3_1 , 'conv3_2')
    conv3_3 = conv_layer(conv3_2 , 'conv3_3')
    pool3 = max_pool(conv3_3 , 'pool3')
    
    conv4_1 = conv_layer(pool3 , 'conv4_1')
    conv4_2 = conv_layer(conv4_1 , 'conv4_2')
    conv4_3 = conv_layer(conv4_2 , 'conv4_3')
    pool4 = max_pool(conv4_3 , 'pool4')
    
    conv5_1 = conv_layer(pool4 , 'conv5_1')
    conv5_2 = conv_layer(conv5_1 , 'conv5_2')
    conv5_3 = conv_layer(conv5_2 , 'conv5_3')
    pool5 = max_pool(conv5_3 , 'pool5')
    flatten = tf.reshape(pool5 , [-1 , 7 * 7 * 512])


# fully connected layer，這部分就是直接重新訓練了
with tf.variable_scope('fully_connected_layer'): 
    fc6 = tf.layers.dense(flatten , 256 , tf.nn.relu , name = 'fc6')
    prediction = tf.layers.dense(fc6 , 1 , name = 'out')


loss = tf.losses.mean_squared_error(labels = output_y , predictions = prediction)

# 決定fine_tune與main_train的variable
# 因為fine_tune的learning rate會比main_train來的小，且限制fine_tune部分的梯度大小
# fine_tune就是'vgg16.npy'中訓練好的weight與bias
# main_train就是fully connected layer的weight與bias
var_fine_tune = [var for var in tf.trainable_variables() if 'fully_connected_layer' not in var.name]
var_main_train = [var for var in tf.trainable_variables() if 'fully_connected_layer' in var.name]

# fine_tune部分
optimizer_fine_tune = tf.train.RMSPropOptimizer(0.0001)
gradients = optimizer_fine_tune.compute_gradients(loss , var_list = var_fine_tune)
train_op_fine_tune = optimizer_fine_tune.apply_gradients(gradients)

# main_train部分
train_op_main_train = tf.train.RMSPropOptimizer(0.001).minimize(loss , var_list = var_main_train)

# 將fine_tune與main_train兩部分的train_op合併在一起
train_op = tf.group(train_op_fine_tune , train_op_main_train)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

# minibatch data index
epochs = 60
batch_size = 10
step = (math.ceil(len(xs) / batch_size)) * batch_size
temp = []
j = 0
index = []
for ii in range(0 , step):
    j = j + 1
    if j > len(xs):
        j = j - len(xs)   
    temp.append(j)  
    if len(temp) == batch_size:
       index.append(temp)
       temp = []
index = list(np.array(index) - 1)

for epoch_i in range(0 , epochs):
    shuffle = np.arange(len(xs))
    np.random.shuffle(shuffle)
    xs = xs[shuffle]
    ys = ys[shuffle]
    
    for batch_i in range(0 , 50):
        _ , loss_ = sess.run([train_op , loss] , feed_dict = {input_x : xs[index[batch_i]] , output_y : ys[index[batch_i]]})
        print('epoch : {} , batch : {} , loss : {:.2f}'.format(epoch_i , batch_i , loss_))

# 存取model
saver = tf.train.Saver()
saver.save(sess , 'model/transfer_learn')
