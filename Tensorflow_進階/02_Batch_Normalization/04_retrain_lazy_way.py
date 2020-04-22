import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

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


#----------------------model restore----------------------# 
sess = tf.Session()
new_saver = tf.train.import_meta_graph(os.path.join('my_network/save_net.meta'))
new_saver.restore(sess, tf.train.latest_checkpoint(os.path.join('my_network')))
#----------------------model restore----------------------# 


#--------------model parameter saved before--------------#
graph = tf.get_default_graph()
xs = graph.get_tensor_by_name('xs:0')
ys = graph.get_tensor_by_name('ys:0')
on_train = graph.get_tensor_by_name('on_train:0')           
cost = graph.get_tensor_by_name('cost:0')
prediction =  graph.get_tensor_by_name('prediction/output:0')
train_op_retrain = tf.get_collection('new_way')[0]
#--------------model parameter saved before--------------#


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
        sess.run(train_op_retrain , feed_dict = {xs : x_data[index[i]] , ys : y_data[index[i]] , on_train: True})           
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
        
# save model
saver = tf.train.Saver(var_list = train_variable_list)        
saver.save(sess , 'my_network_retrain_lazy_way/save_net_retrain_lazy_way')        
            
