"""
lstm cell is divided into two parts (主線_state & 支線_state)        
state[0] => 主線
state[1] => 支線  
       
     state[1] : 也同時是每個cell的output
        ↑
  ╔═══════════╗     
  ║           ║      
  ║ lstm cell ║  → state[0] : 要傳入下一個時間點的lstm cell
  ║           ║
  ╚═══════════╝          
"""  

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
batch_size = 100

n_inputs = 28   # MNIST data input (img shape: 28*28)
n_steps = 28    # time steps
n_hidden_units = 128   # neurons in hidden layer
n_classes = 10      # MNIST classes (0-9 digits)
layer_num = 1

# tf Graph input
x = tf.placeholder(tf.float32 , [None , n_steps, n_inputs])
y = tf.placeholder(tf.float32 , [None , n_classes])

# Define weights
weights = {'in': tf.Variable(tf.random_normal([n_inputs , n_hidden_units])),    # (28, 128)
           'out': tf.Variable(tf.random_normal([n_hidden_units , n_classes]))}  # (128, 10)
    
biases = {'in': tf.Variable(tf.constant(0.1 , shape = [1 , n_hidden_units])),   # (1 , 128)
          'out': tf.Variable(tf.constant(0.1 , shape = [1 , n_classes]))}       # (1 , 10)


#---------hidden layer for input to cell---------#
# transpose the inputs shape from
# X ==> (100 batch * 28 steps , 28 inputs)
X = tf.reshape(x , [-1 , n_inputs])

# into hidden
# X_in_temp = (100 batch * 28 steps , 128 hidden)
X_in_temp = tf.matmul(X , weights['in']) + biases['in']

# X_in ==> (100 batch, 28 steps, 128 hidden => 
#           現在總共有100筆圖片，每張圖片會有28個時間點(row)，每個時間點(row)會有128個element，
#           以第一個cell為例，送進第 1 個cell的資料維100張圖片的第 1 個時間點(row)所集結起來的資料
#           送進第 2 個cell的資料維100張圖片的第 2 個時間點(row)所集結起來的資料) 
X_in = tf.reshape(X_in_temp , [-1 , n_steps , n_hidden_units])
#---------hidden layer for input to cell---------#


cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units , forget_bias = 1.0 , state_is_tuple = True)
mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layer_num , state_is_tuple = True) 
# 使用tf.nn.rnn_cell.MultiRNNCell(多層 RNN)的話，記得restart kernel

init_state = mlstm_cell.zero_state(batch_size , dtype = tf.float32)
state = init_state
outputs , states = [] , []
with tf.variable_scope('LSTM'):
    for timestep in range(0 , n_steps):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        mlstm_cell_output , state = mlstm_cell(X_in[: , timestep , :] , state)
        outputs.append(mlstm_cell_output) 
        states.append(state)
results = tf.matmul(outputs[-1] , weights['out']) + biases['out']    # shape = (128, 10)

#output , state  = tf.nn.dynamic_rnn(mlstm_cell , 
#                                    inputs = X_in, 
#                                    initial_state = init_state, 
#                                    time_major = False)
#results = tf.matmul(output[: , -1 , :] , weights['out']) + biases['out']    # shape = (128, 10)    

pred = results
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred , labels = y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct = tf.equal(tf.argmax(pred , 1) , tf.argmax(y , 1))
correct = tf.cast(correct , tf.float32)
accuracy = tf.reduce_mean(correct)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
  
for step in range(0 , 1000):   
    batch_xs , batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape([batch_size , n_steps, n_inputs])
    
    if step == 0:
        feed_dict = {x : batch_xs , y : batch_ys}
    elif step > 0:
        feed_dict = {x : batch_xs , y : batch_ys , init_state : state_current}
    
    sess.run(train_op , feed_dict)    
    state_current = sess.run(states[-1] , feed_dict)
    
    if step % 20 == 0:
        train_accuracy = sess.run(accuracy , feed_dict)
        print('step {} : {:.2%}'.format(step , train_accuracy))


# 看一下每個time_step的最終輸出結果的變化
import matplotlib.pyplot as plt
visualization_xs , _ = mnist.train.next_batch(batch_size)
visualization_xs = visualization_xs.reshape([-1 , 28, 28])
# 查看第1張圖是什麼
plt.imshow(visualization_xs[1 , : , :] , cmap = 'gray')  
visualization_output = sess.run(outputs , feed_dict = {x : visualization_xs})

bar_index = range(10)
plt.figure(figsize = (30 , 25))
plt.subplots_adjust(hspace = 0.5)
for time_step in range(0 , 28):
    plt.subplot(7 , 4 , time_step + 1)
    time_step_output = tf.matmul(visualization_output[time_step] , weights['out']) + biases['out']
    pro = sess.run(tf.nn.softmax(time_step_output))
    # 查看第1張圖在每一個timestep機率分布的變化
    plt.bar(bar_index , pro[1 , :] , width = 0.4 , align = 'center') 
    plt.xticks(bar_index)
    plt.show()