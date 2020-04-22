import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# load data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#-------------setting neuron network structure-------------#
def conv_layer(inputs , shape1 , shape2 , activaction_function = None):
    weight = tf.Variable(tf.truncated_normal(shape1 , stddev = 0.1))
    bias = tf.Variable(tf.constant(0.1 , shape = shape2))
    conv_w_plus_b = tf.nn.conv2d(inputs , weight , strides = [1 , 1 , 1 , 1] , padding = 'SAME') + bias
    if activaction_function is None:
        outputs = conv_w_plus_b
    else:
        outputs = activaction_function(conv_w_plus_b)
    return outputs   
    
def max_pool_2x2(x):
	return tf.nn.max_pool(x , ksize = [1 , 2 , 2 , 1] , strides = [1 , 2 , 2 , 1] , padding = 'SAME')    

def fc_layer(inputs , in_size , out_size , activaction_function = None):
    Weights = tf.Variable(tf.truncated_normal([in_size , out_size] , stddev = 0.1))
    biases = tf.Variable(tf.zeros([1 , out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs , Weights) + biases	
    if activaction_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activaction_function(Wx_plus_b)
    return outputs



# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32 , [None , 784]) / 25
ys = tf.placeholder(tf.float32 , [None , 10])
x_image = tf.reshape(xs , [-1 , 28 , 28 , 1])


## conv1 layer ##
h_conv1 = conv_layer(x_image , [5 , 5 , 1 , 32] , [1 , 32] , activaction_function = tf.nn.relu)
h_pool1 = max_pool_2x2(h_conv1) 

## conv2 layer ##
h_conv2 = conv_layer(h_pool1 , [5 , 5 , 32 , 64] , [1 , 64] , activaction_function = tf.nn.relu)
h_pool2 = max_pool_2x2(h_conv2)

## fully connected layer 1 ##
h_pool2_flat = tf.reshape(h_pool2 , [-1 , 7 * 7 * 64])
h_fc1 = fc_layer(h_pool2_flat , 7 * 7 * 64 , 1024 , activaction_function = tf.nn.relu)
h_fc1_drop = tf.nn.dropout(h_fc1 , keep_prob)

## fully connected layer 2 ##
prediction = fc_layer(h_fc1_drop , 1024 , 10 , activaction_function = tf.nn.softmax)
prediction = tf.log(prediction + 1e-9)

# the error between prediction and real data
cross_entropy_temp = -tf.reduce_sum(ys * prediction , axis = 1)
cross_entropy = tf.reduce_mean(cross_entropy_temp)   
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#-------------setting neuron network structure-------------#


#-------------starting to train neuron network-------------# 
sess = tf.Session() 
sess.run(tf.global_variables_initializer())

for i in range(1000):
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

test = sess.run(prediction , feed_dict = {xs : batch_xs , ys: batch_ys , keep_prob: 1})

