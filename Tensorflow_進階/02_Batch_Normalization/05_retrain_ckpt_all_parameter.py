import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

'''
這支程式只是把之前訓練好的模型調出來，再丟新的數據進去訓練全部或部分的參數
無法做到加新的layer，再去訓練新的layer的參數，也就是要達到transfer learning的效果
若要達到上述效果可以參考 資料夾「07_Vrious_Application」中的
資料夾「transfer_learning」的 step_1_save_model.py 與 step_2_add_new_layer.py
'''

# make up data
x_data = np.linspace(-10 , 20 , 2500).reshape([-1 , 1])
np.random.shuffle(x_data)
noise = np.random.normal(0 , 8 , x_data.shape)
y_data = np.square(x_data) - 5 + noise


#----------------------original network----------------------#
def batch_norm_layer(inputs , on_train):
    # the dimension you wanna normalize, here [0] for batch
    fc_mean , fc_var = tf.nn.moments(inputs , axes = [0] , name = 'mean_var')
    
    ema = tf.train.ExponentialMovingAverage(decay = 0.5)
    ema_apply_op = ema.apply([fc_mean , fc_var])
    mean = tf.cond(on_train , lambda : fc_mean , lambda : ema.average(fc_mean))
    var = tf.cond(on_train , lambda : fc_var , lambda : ema.average(fc_var))
    
    scale = tf.Variable(tf.ones([1 , inputs.shape[1].value]) , name = 'scale')
    shift = tf.Variable(tf.zeros([1 , inputs.shape[1].value]) , name = 'shift')
                                
    temp = (inputs - mean) / tf.sqrt(var + 0.001)
    outputs = tf.multiply(temp , scale) + shift
  
    return outputs , ema_apply_op 
    

def add_layer(inputs , in_size , out_size):
    # weights and biases (bad initialization for this case)
    Weights = tf.Variable(tf.random_normal([in_size , out_size] , mean = 0. , stddev = 1.) , name = 'weight')
    biases = tf.Variable(tf.zeros([1 , out_size]) + 0.1 , name = 'bias')

    # fully connected product
    Wx_plus_b = tf.add(tf.matmul(inputs , Weights) , biases , name = 'output')
    outputs = Wx_plus_b
    
    return outputs
    
xs = tf.placeholder(tf.float32 , [None , 1] , name = 'xs')  # [num_samples , num_features]
ys = tf.placeholder(tf.float32 , [None , 1] , name = 'ys')
on_train = tf.placeholder(tf.bool , name = 'on_train') # train/test selector for batch normalisation

# record input
input_ = []
input_.append(xs)
ema_ = []
           
# build hidden layers & batch normalization layers
N_HIDDEN_UNITS = 30
for l_n in range(0 , 7):
    with tf.variable_scope('layer_' + str(l_n + 1)): 
        L = add_layer(input_[l_n] , input_[l_n].shape[1].value , N_HIDDEN_UNITS)
    
    with tf.variable_scope('layer_norm_' + str(l_n + 1)):
        L_norm , ema = batch_norm_layer(L , on_train)
        #L_norm = batch_norm_layer(L , on_train = tf.constant(True, dtype=tf.bool))
    
    L_act = tf.nn.tanh(L_norm , name = 'layer_act_' + str(l_n + 1))    
    input_.append(L_act)
    ema_.append(ema) 

update_ema = tf.group(ema_)    
 
# build output layer
with tf.variable_scope('prediction'):    
    prediction = add_layer(input_[-1] , input_[-1].shape[1].value , 1)
    
cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction) , reduction_indices = [1]) , name = 'cost')
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
#----------------------original network----------------------# 


#----------------------model restore----------------------# 
sess = tf.Session()
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(os.path.join('my_network')) # 獲取checkpoints對象
if ckpt and ckpt.model_checkpoint_path: #判斷ckpt是否為空，若不為空，才進行模型的加载，否則從頭開始訓練  
    saver.restore(sess , ckpt.model_checkpoint_path) #恢復保存的網路結構，實現斷點續訓 
# 用這樣的方式把讀取，需要將原來的network再建立一次，比較麻煩
#----------------------model restore----------------------# 


#----------------------retrain model----------------------#
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
index = minibatch_index(minibatch_num = 100 , data = x_data)

# record cost
cost_his = []

record_step = 5
epochs = 100
for k in range(0 , epochs):
    for i in range(0 , len(index)):
        # train on batch
        sess.run([train_op , update_ema] , feed_dict = {xs : x_data[index[i]] , ys : y_data[index[i]] , on_train: True})           
        if i % record_step == 0:
            # record cost
            cost_his.append(sess.run(cost , feed_dict = {xs : x_data , ys : y_data , on_train : True}))
#----------------------retrain model----------------------#
 
       
# plot cost curve         
plt.ioff()
plt.figure()
plt.plot(np.arange(len(cost_his)) , np.array(cost_his) , label = 'with BN')     
plt.legend()
plt.show()

# plot prediction curve
plt.ioff()
plt.figure()
x_test = np.linspace(-10 , 20 , 2500).reshape([-1 , 1])
y_test = np.square(x_test) - 5 + np.random.normal(0 , 8 , x_data.shape)
plt.scatter(x_test , y_test)
pred = sess.run(prediction , feed_dict = {xs : x_test , ys : y_test , on_train : False})
plt.plot(x_test , pred , 'r')
plt.show()

# record variable list
train_variable_list = tf.trainable_variables()
all_list = tf.global_variables()
bn_mean_var = []
for g in range(0 , len(all_list)):
    if 'mean_var' in all_list[g].name:
        train_variable_list.append(all_list[g])

# save retrain model
retrain_saver = tf.train.Saver(var_list = train_variable_list)        
retrain_saver.save(sess , 'my_network_retrain/save_net_retrain')
