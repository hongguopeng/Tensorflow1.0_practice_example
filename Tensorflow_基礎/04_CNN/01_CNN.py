import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# load data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

#-------------setting neuron network structure-------------#
def weight_variable(shape):
    initial = tf.truncated_normal(shape , stddev = 0.1) # stddev => standard deviation
    #	initial = tf.random_normal(shape)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1 , shape = shape)
    return tf.Variable(initial)

"""
1. padding = 'SAME'  => outputsize = input_size / stride
                        outputsize 若非整數則無條件進位
2. padding = 'VALID' => outputsize = (input_size - kernel_size + 1) / stride
                        outputsize 若非整數則無條件進位                
"""

def conv2d(x , W):
    # stride [1, x_movement , y_movement, 1]
    # Must have strides[0] = strides[3] = 1	
    return tf.nn.conv2d(x , W , strides = [1 , 1 , 1 , 1] , padding = 'SAME') 

def max_pool_2x2(x):
    # 不需要跟tf.nn.conv2d一樣要輸入W
    # ksize = [1 , *2* , *2* , 1] 輸入 2 , 2 代表每2x2個pixel做一次選取pixel最大的動作
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x , ksize = [1 , 2 , 2 , 1] , strides = [1 , 2 , 2 , 1] , padding = 'SAME')

# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32 , [None , 784]) / 25
ys = tf.placeholder(tf.float32 , [None , 10])
#print(x_image.shape) # [n_sample , 28 , 28 , 1]
x_image = tf.reshape(xs , [-1 , 28 , 28 , 1]) # 最後一個1代表channel，黑白為1，彩色為3


## conv1 layer ##
W_conv1 = weight_variable([5 , 5 , 1 , 32]) # filter為5x5的size，1張原圖片可產生32張圖片
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image , W_conv1) + b_conv1) # output size : 28x28x32
h_pool1 = max_pool_2x2(h_conv1) # output size : 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5 , 5 , 32 , 64]) # filter為5x5的size，現在總共產生64張圖片 ps:不是"32張圖片每張圖片產生64張"，是總共64張
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1 , W_conv2) + b_conv2) # output size : 14x14x64
h_pool2 = max_pool_2x2(h_conv2) # output size : 7x7x64

## fully connected layer 1 ##
W_fc1 = weight_variable([7 * 7 * 64 , 1024])
b_fcl = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2 , [-1 , 7 * 7 * 64]) # [n_samples , 7 , 7 , 64] => [n_samples , 7 * 7 * 64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat , W_fc1) + b_fcl)
h_fc1_drop = tf.nn.dropout(h_fc1 , keep_prob)

## fully connected layer 2 ##
W_fc2 = weight_variable([1024 , 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop , W_fc2) + b_fc2)
prediction = tf.log(prediction + 1e-9)

# the error between prediction and real data
cross_entropy_temp = -tf.reduce_sum(ys * prediction , axis = 1)
cross_entropy = tf.reduce_mean(cross_entropy_temp)    
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#-------------setting neuron network structure-------------#


#-------------starting to train neuron network-------------# 
sess = tf.Session() 
sess.run(tf.global_variables_initializer())

for i in range(0 , 1000):
    batch_xs , batch_ys = mnist.train.next_batch(100)
    sess.run(train_step , feed_dict = {xs : batch_xs , ys: batch_ys , keep_prob: 0.5})
    print(sess.run(cross_entropy , feed_dict = {xs : batch_xs , ys: batch_ys , keep_prob: 0.5}))

#-------------starting to train neuron network-------------#    
   

# accuracy of data_set   
def data_accuracy(X_data , y_data):
    count = 0
    for k in range(0 , len(X_data) - 1):
        test_pre = sess.run(prediction , feed_dict = {xs: X_data[k : k + 1 , :] , keep_prob : 1})
        if np.argmax(test_pre , 1) == np.argmax(y_data[k : k + 1 , :] , 1):
            count = count + 1
    accuracy = count / len(X_data)
    return accuracy
print('accuracy:' , data_accuracy( mnist.test.images[0 : 1000] , mnist.test.labels[0 : 1000]))

test = sess.run(h_fc1 , feed_dict = {xs : batch_xs , ys: batch_ys , keep_prob: 1})


