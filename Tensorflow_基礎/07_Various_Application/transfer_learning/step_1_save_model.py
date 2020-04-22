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
        outputs = activaction_function(Wx_plus_b , name = 'output')
    
    return outputs	

def output_layer(inputs , in_size , out_size , activaction_function = None):
    Weights = tf.Variable(tf.truncated_normal([in_size , out_size] , mean = 0.01 , stddev = 0.1) , name = 'w')
    biases = tf.Variable(tf.zeros([1 , out_size]) + 0.01 , name = 'b')
    Wx_plus_b = tf.matmul(inputs , Weights) + biases

    if activaction_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activaction_function(Wx_plus_b , name = 'output')
        
    return outputs 


# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32 , name = 'keep_prob')
xs = tf.placeholder(tf.float32 , [None , 64] , name = 'xs')
ys = tf.placeholder(tf.float32 , [None , 10] , name = 'ys')


# add hidden layer
with tf.variable_scope('layer_1'): 
    layer_1 = add_layer(xs , 64 , 100 , activaction_function = tf.nn.relu)
with tf.variable_scope('layer_2'): 
    layer_2 = add_layer(layer_1 , 100 , 50 , activaction_function = tf.nn.tanh)

# add output layer
with tf.variable_scope('output_layer'): 
    prediction = output_layer(layer_2 , 50 , 10 , activaction_function = tf.nn.softmax)


# the error between prediction and real data
cross_entropy_temp = -tf.reduce_sum(ys * tf.log(prediction) , 1)
cross_entropy = tf.reduce_mean(cross_entropy_temp)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# save model
saver = tf.train.Saver()

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
num = 100
step = (int(len(X_train) / num) + 1) * num
temp = []
j = 0
index = []
for ii in range(0 , step):
    j = j + 1
    if j > len(X_train):
        j = j - (len(X_train))   
    temp.append(j)  
    if len(temp) == num:
       index.append(temp)
       temp = []
index = list(np.array(index) - 1)

shuffle = 1
shuffle_data = shuffle_function()
if shuffle == 1:
    X_data , y_data = shuffle_data[0] , shuffle_data[1]
elif shuffle == 0:
    X_data , y_data = X_train , y_train
    
    
train_cs , val_cs = [] , []
for i in range(0 , epochs):        
    for m in range(0 , len(index)):        
        sess.run(train_step , feed_dict = {xs : X_data[index[m] , :] , 
                                           ys : y_data[index[m] , :] , 
                                           keep_prob : 0.5})
    if i % 25 == 0:
        train_loss = sess.run(cross_entropy , feed_dict = {xs : X_train , ys : y_train , keep_prob : 1}) 	  
        train_cs.append(train_loss.astype(np.float32))
        val_loss = sess.run(cross_entropy , feed_dict = {xs: X_val , ys : y_val , keep_prob : 1}) 	
        val_cs.append(val_loss.astype(np.float32))
        
saver.save(sess , 'my_network/save_net')

#-------------starting to train neuron network-------------#
    
    

   
# cross entropy curve	
plt.figure(figsize=(6 , 3))
plt.plot(train_cs , 'g-*')
plt.plot(val_cs , 'r-o')	


# accuracy of data_set
def data_accuracy(predict , X_data , y_data):
    count = 0
    for k in range(0 , len(X_data)):
        test_pre = sess.run(predict , feed_dict = {xs: X_data[k : k + 1 , :] , keep_prob : 1})
        if np.argmax(test_pre , 1) == np.argmax(y_data[k : k + 1 , :] , 1):
            count = count + 1
    accuracy = count / len(X_data)
    return accuracy 
print('accuracy:' , data_accuracy(prediction , X_test , y_test))   


test_pre = sess.run(prediction , feed_dict = {xs: X_test , keep_prob : 1}) 
pre = np.argmax(test_pre , 1) 
true_label = np.argmax(y_test , 1)
test_data_accuracy = (pre == true_label).astype('float32').sum() / len(X_test)

# middle layer output 
layer_2_output = sess.run(layer_2 , feed_dict = {xs: X_test , keep_prob : 1})    
        



	   