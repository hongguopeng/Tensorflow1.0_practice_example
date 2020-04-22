import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer , OneHotEncoder
import os
import matplotlib.pyplot as plt

'''這支程式是能加新的layer去訓練，並不是單純將就存檔的模型調出來再丟新的數據訓練參數'''

#------------------------load and organize data------------------------#
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2)
X_train , X_val , y_train , y_val = train_test_split(X_train , y_train , test_size = 0.2)

def y_conversion_format(y):
    y_ = np.zeros([y.shape[0] , ])
    for i in range(0 , len(y_)):
        count = 0
        while True:
            if y[i][count] == 0: count = count + 1
            elif y[i][count] == 1: break
        y_[i] = count
    return y_

# 把label為0與1的data挑出來，再用transfer learning的方式去預測這些data(現在只有兩個class)
y_train_ = y_conversion_format(y_train).astype(np.int32)
y_test_ = y_conversion_format(y_test).astype(np.int32)
y_val_ = y_conversion_format(y_val).astype(np.int32)
X_train_01 , y_train_01 = X_train[y_train_ <= 1] , y_train_[y_train_ <= 1]  
X_test_01 , y_test_01 = X_test[y_test_ <= 1] , y_test_[y_test_ <= 1]  
X_val_01 , y_val_01 = X_val[y_val_ <= 1] , y_val_[y_val_ <= 1]  

onehotencoder = OneHotEncoder(categorical_features = [0])
y_train_01 = onehotencoder.fit_transform(pd.DataFrame(y_train_01)).toarray()
y_test_01 = onehotencoder.fit_transform(pd.DataFrame(y_test_01)).toarray()
y_val_01 = onehotencoder.fit_transform(pd.DataFrame(y_val_01)).toarray()
#------------------------load and organize data------------------------#


#------------------------restore model------------------------#
sess = tf.Session()
saver = tf.train.import_meta_graph(os.path.join('my_network/save_net.meta'))
saver.restore(sess, tf.train.latest_checkpoint(os.path.join('my_network')))
#------------------------restore model------------------------#


#------------------------model parameter saved before------------------------#
graph = tf.get_default_graph()
xs = graph.get_tensor_by_name('xs:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')
output_last = graph.get_tensor_by_name('layer_2/output:0') 
#------------------------model parameter saved before------------------------#


#------------------------new layer and parameter------------------------#
ys = tf.placeholder(tf.float32 , [None , 2] , name = 'ys') 

def add_layer(inputs , in_size , out_size , activaction_function = None):
    Weights = tf.Variable(tf.truncated_normal([in_size , out_size] , mean = 0.01 , stddev = 0.1) , name = 'w')
    biases = tf.Variable(tf.zeros([1 , out_size]) + 0.01 , name = 'b')
    Wx_plus_b = tf.matmul(inputs , Weights) + biases

    if activaction_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activaction_function(Wx_plus_b , name = 'output')
        
    return outputs 

with tf.variable_scope('new_layer'): 
    new_layer_output = add_layer(output_last , 50 , 2 , activaction_function = tf.nn.softmax)
#------------------------new layer and parameter------------------------#
  
    
#------------------------compute cost------------------------#    
with tf.variable_scope('cost'):    
    cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - new_layer_output) , axis = 1) , name = 'cross_entropy')
#------------------------compute cost------------------------#    


#------------------------training setting------------------------#
# 只訓練'new_layer'下的'w'與'b'，其他參數在之前已經訓練好了，所以將之固定起來不參與訓練
with tf.variable_scope('train'):      
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost , var_list = tf.trainable_variables()[-2:])    
#------------------------training setting------------------------#


#------------------------initialize parameter------------------------#
# 因為只訓練'new_layer'下的 w 與 b，所以在初始化參數時，只需初始化'new_layer'下的'w'與'b'
# tf.initialize_variables(初始化的參數) 
init_part_variable = tf.initialize_variables(tf.trainable_variables()[-2:])
sess.run(init_part_variable)
#------------------------initialize parameter------------------------#


#------------------------start training------------------------# 
def minibatch_index(minibatch_num , data):
    step = (int(len(data) / minibatch_num) + 1) * minibatch_num
    temp = []
    index = []
    j = 0
    for ii in range(0 , step):
        j = j + 1
        if j > len(data):
            j = j - (len(data))   
        temp.append(j)  
        if len(temp) == minibatch_num:
           index.append(temp)
           temp = []
    index = list(np.array(index) - 1) 
    return index
index = minibatch_index(minibatch_num = 10 , data = X_train_01)

# record cost
cost_his = []

record_step = 1
epochs = 100
for k in range(0 , epochs):
    for i in range(0 , len(index)):
        # train on batch
        sess.run(train_op , feed_dict = {xs : X_train_01[index[i]] , ys : y_train_01[index[i]] , keep_prob : 1})           
        if i % record_step == 0:
            cost_his.append(sess.run(cost , feed_dict = {xs : X_val_01 , ys : y_val_01 , keep_prob : 1}))    
plt.plot(cost_his)
#------------------------start training------------------------# 


def data_accuracy(predict , X_data , y_data):
    test_pre = sess.run(predict , feed_dict = {xs: X_data , keep_prob : 1})
    count = np.sum(np.argmax(test_pre , 1) == np.argmax(y_data , 1))
    accuracy = count / len(X_data)
    return accuracy
print('accuracy:' , data_accuracy(new_layer_output , X_test_01 , y_test_01)) 