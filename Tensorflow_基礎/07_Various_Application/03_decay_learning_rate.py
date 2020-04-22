import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import random
import math
from tqdm import trange
import matplotlib.pyplot as plt


# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2)
X_train , X_val , y_train , y_val = train_test_split(X_train , y_train , test_size = 0.2)

#-------------setting neuron network structure-------------#
def add_layer(inputs , in_size , out_size , activaction_function = None):
    Weights = tf.Variable(tf.truncated_normal([in_size , out_size] , mean = 0.01 , stddev = 0.1))
    biases = tf.Variable(tf.zeros([1 , out_size]) + 0.1)
    Wx_plus_b = tf.add(tf.matmul(inputs , Weights) , biases) 
	
	  # here to dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b , keep_prob)
	   
    # here to activaction_function
    if activaction_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activaction_function(Wx_plus_b)
        
    return outputs

def output_layer(inputs , in_size , out_size , activaction_function = None):
    Weights = tf.Variable(tf.truncated_normal([in_size , out_size] , mean = 0.01 , stddev = 0.1))
    biases = tf.Variable(tf.zeros([1 , out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs , Weights) + biases
 
    # here to activaction_function
    if activaction_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activaction_function(Wx_plus_b)

    return outputs	


# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32 , [None , 64])
ys = tf.placeholder(tf.float32 , [None , 10])


# add hidden layer
layer_1 = add_layer(xs , 64 , 100 , activaction_function = tf.nn.relu )
layer_2 = add_layer(layer_1 , 100 , 50 , activaction_function = tf.nn.tanh)

# add output layer
prediction = output_layer(layer_2 , 50 , 10 , activaction_function = tf.nn.softmax)
prediction = tf.log(prediction + 1e-9)

# the error between prediction and real data
cross_entropy_temp = -tf.reduce_sum(ys * prediction , axis = 1)
cross_entropy = tf.reduce_mean(cross_entropy_temp)
correct = tf.equal(tf.argmax(prediction , 1) , tf.argmax(ys , 1))
correct = tf.cast(correct , tf.float32)
accuracy = tf.reduce_mean(correct)

global_ = tf.Variable(0 , trainable = False)
# 在這個例子中，batch_size為100，所以decay_steps=math.ceil(len(X_train) / 100)代表每訓練一個batch的過程中
# (也可以說是在一個epoch的過程中)，learning_rate會維持平坦，訓練一個batch後，learning_rate才會往下掉 
l_r = tf.train.exponential_decay(0.1 , global_ , decay_steps = math.ceil(len(X_train) / 100) , decay_rate = 0.9 , staircase = True)    
train_step = tf.train.GradientDescentOptimizer(l_r).minimize(cross_entropy , global_step = global_)
#-------------setting neuron network structure-------------#


#-------------starting to train neuron network-------------#
sess = tf.Session() 
sess.run(tf.global_variables_initializer())


# shuffle X_train -- way1
#def shuffle_function():
#    X_train_shuffle , y_train_shuffle = [] , []
#    X_train_temp , y_train_temp = list(X_train) , list(y_train)
#    for k in range(0 , len(X_train)):
#        takeout_index = random.randint(0 , len(X_train_temp) - 1)
#        takeout_x , takeout_y = X_train_temp[takeout_index] , y_train_temp[takeout_index]
#        X_train_shuffle.append(takeout_x) , y_train_shuffle.append(takeout_y)
#        del X_train_temp[takeout_index] , y_train_temp[takeout_index]
#    return [np.array(X_train_shuffle) , np.array(y_train_shuffle)]

# shuffle X_train -- way2
def shuffle_function():
    shuffle_indices = np.random.permutation(np.arange(len(X_train)))
    X_train_shuffle , y_train_shuffle = X_train[shuffle_indices] , y_train[shuffle_indices]
    return [np.array(X_train_shuffle) , np.array(y_train_shuffle)]


# minibatch data index
epochs = 1000
batch_num = 100
step = (math.ceil(len(X_train) / batch_num)) * batch_num
temp = []
j = 0
index = []
for ii in range(0 , step):
    j = j + 1
    if j > len(X_train):
        j = j - (len(X_train))   
    temp.append(j)  
    if len(temp) == batch_num:
       index.append(temp)
       temp = []
index = list(np.array(index) - 1)

shuffle = 1
shuffle_data = shuffle_function()
if shuffle == 1:
    X_data , y_data = shuffle_data[0] , shuffle_data[1]
elif shuffle == 0:
    X_data , y_data = X_train , y_train
    
    
train_cs , val_cs , train_acc , val_acc = [] , [] , [] , []
learning_rate = []
for i in trange(0 , epochs):        
    for m in range(0 , len(index)):        
        sess.run(train_step , feed_dict = {xs : X_data[index[m] , :] , 
                                           ys : y_data[index[m] , :] , 
                                           keep_prob : 0.5})
        print('learnng rate:%.8f | global_step:%d' %(sess.run(l_r) , sess.run(global_)) )
        learning_rate.append(sess.run(l_r))
    if i % 25 == 0:
        train_loss , train_accuracy = sess.run([cross_entropy , accuracy] , feed_dict = {xs: X_train , ys : y_train , keep_prob : 1}) 	  
        train_cs.append(train_loss)
        train_acc.append(train_accuracy)
        
        val_loss , val_accuracy = sess.run([cross_entropy , accuracy] , feed_dict = {xs: X_val , ys : y_val , keep_prob : 1}) 	
        val_cs.append(val_loss)		
        val_acc.append(val_accuracy)
#-------------starting to train neuron network-------------#       

        
# cross entropy curve	
plt.figure(figsize=(6 , 3))
plt.plot(train_cs , 'g-*')
plt.plot(val_cs , 'r-o')	
plt.figure()
plt.plot(learning_rate)


# accuracy of data_set
test_acc = sess.run(accuracy , feed_dict = {xs : X_test , ys : y_test , keep_prob : 1})
print('test accuracy:' , test_acc)  