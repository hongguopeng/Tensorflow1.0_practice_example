import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
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
    Weights = tf.Variable(tf.truncated_normal([in_size , out_size] , mean = 0.01 , stddev = 0.1) , name = 'w')
    biases = tf.Variable(tf.zeros([1 , out_size]) + 0.01 , name = 'b')
    Wx_plus_b = tf.matmul(inputs , Weights) + biases
	
	  # here to dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b , keep_prob)
	
    if activaction_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activaction_function(Wx_plus_b)
    
    return outputs	

def output_layer(inputs , in_size , out_size , activaction_function = None):
    Weights = tf.Variable(tf.truncated_normal([in_size , out_size] , mean = 0.01 , stddev = 0.1) , name = 'w')
    biases = tf.Variable(tf.zeros([1 , out_size]) + 0.01 , name = 'b')
    Wx_plus_b = tf.matmul(inputs , Weights) + biases

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
with tf.variable_scope('hidden_layer'):
    layer_1 = add_layer(xs , 64 , 50  , activaction_function = tf.nn.relu)
    
# add output layer
with tf.variable_scope('output_layer'):
    prediction = output_layer(layer_1 , 50 , 10 , activaction_function = tf.nn.softmax)


# the error between prediction and real data
# 在出現nan值的loss中一般是使用TensorFlow的log函数，然後計算得到的Nan
# 在這個例子中layer_1使用了relu的activaction_function，因此最後在prediction
# 會有出現0的機會，因此使用tf.clip_by_value將prediction最小的值換成1e-8
# 如此在算cross_entropy時，就不會出現nan值  
# tf.clip_by_value(欲cilp的值 , 最小值 , 最大值)                                                                
solve_nan_prob = tf.log(tf.clip_by_value(prediction , 1e-8 , tf.reduce_max(prediction)))    
cross_entropy_temp = -tf.reduce_sum(ys * solve_nan_prob , 1)
with tf.variable_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(cross_entropy_temp)
    correct = tf.equal(tf.argmax(prediction , 1) , tf.argmax(ys , 1))
    correct = tf.cast(correct , tf.float32)
    accuracy = tf.reduce_mean(correct)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#-------------setting neuron network structure-------------#



#-------------starting to train neuron network-------------#
sess = tf.Session() 
sess.run(tf.global_variables_initializer())

# train model
train_cs , val_cs , train_acc , val_acc = [] , [] , [] , []
for step in range(0 , 1000):
    sess.run(train_step , feed_dict = {xs : X_train , ys : y_train , keep_prob : 0.5})
    if step % 25 == 0:	   
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


# accuracy of data_set
test_acc = sess.run(accuracy , feed_dict = {xs : X_test , ys : y_test , keep_prob : 1})
print('test accuracy:' , test_acc)  