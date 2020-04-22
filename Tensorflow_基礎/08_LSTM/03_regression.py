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
import matplotlib.pyplot as plt

# hyperparameters
lr = 0.006
batch_size = 50
n_inputs = 1   
n_steps = 20
n_hidden_units = 10   
n_outputs = 1
layer_num = 1


# tf Graph input
x = tf.placeholder(tf.float32 , [None , n_steps , n_inputs])
y = tf.placeholder(tf.float32 , [None , n_steps , n_outputs])

# Define weights
weights = {'in': tf.Variable(tf.random_normal([n_inputs , n_hidden_units])),    # (1, 10)  
           'out': tf.Variable(tf.random_normal([n_hidden_units , n_outputs]))}  # (10, 1)    
    
biases = {'in': tf.Variable(tf.constant(0.1 , shape = [1 , n_hidden_units])), # (1 , 10)
          'out': tf.Variable(tf.constant(0.1 , shape = [1 , n_outputs]))}     # (1 , 1)

X = tf.reshape(x , [-1 , n_inputs])

X_in_temp = tf.matmul(X , weights['in']) + biases['in']

X_in = tf.reshape(X_in_temp , [-1 , n_steps , n_hidden_units])

cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units , forget_bias=1.0 , state_is_tuple=True)
mlstm_cell = tf.contrib.rnn.MultiRNNCell([cell] * layer_num , state_is_tuple=True) # layer_num代表有幾層rnn

init_state = mlstm_cell.zero_state(batch_size , dtype = tf.float32)
with tf.variable_scope('LSTM'):
    outputs = []
    for j in range(0 , 50): outputs.append([])   
    state = init_state
    for timestep in range(0 , n_steps):
        if timestep > 0: tf.get_variable_scope().reuse_variables()
        mlstm_cell_output , state = mlstm_cell(X_in[: , timestep , :] , state)
        for k in range(0 , 50):
            outputs[k].append(mlstm_cell_output[k , :]) # 把mlstm_cell_output的第k個batch塞到第k個outputs
    outputs = tf.convert_to_tensor(outputs) # 將 list 轉 tensor
    results = tf.matmul(tf.reshape(outputs , [-1 , n_hidden_units]) , weights['out']) + biases['out'] 

#output , state  = tf.nn.dynamic_rnn(mlstm_cell , 
#                                    inputs = X_in, 
#                                    initial_state = init_state, 
#                                    time_major = False)  
#results = tf.matmul(tf.reshape(output , [-1 , n_hidden_units]) , weights['out']) + biases['out'] 
  
pred = tf.nn.tanh(results)
losses = tf.square(tf.reshape(pred , [-1]) - tf.reshape(y , [-1]))
cost = tf.reduce_mean(losses)
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 每次產生一個1000個點的sin與cos的data，再切分為50個等份(50個batch)，每一個等份20個點(20個time_step)
BATCH_START = 0
TIME_STEPS = n_steps
BATCH_SIZE = batch_size
def get_batch():
    global BATCH_START, TIME_STEPS
    xs = np.arange(BATCH_START , BATCH_START + TIME_STEPS * BATCH_SIZE).reshape((BATCH_SIZE , TIME_STEPS)) / (10 * np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START = BATCH_START + TIME_STEPS
    return [seq[:, :, np.newaxis] , res[:, :, np.newaxis], xs]

plt.ion()   # something about continuous plotting
x_axis , prediction , cost_collect , true_label = [] , [] , [] , []
for i in range(0 , 200):
    seq , res , xs = get_batch()
    
    if i == 0:
        feed_dict = {x : seq , y : res}
    else:
        feed_dict = {x : seq , y : res , init_state : state_current}
                     
    sess.run(train_op , feed_dict)
    state_current = sess.run(state , feed_dict)
    
    cost_collect.append(sess.run(cost , feed_dict))
    
    plt.plot(xs[0 , :] , 
             res[0 , : , :] , 
             c = '#74BCFF' , lw = 1) 
    plt.plot(xs[0 , :] , 
             sess.run(pred , feed_dict).reshape([50 , 20])[0 , :] , 
             c = '#FF9359' , lw = 1)
    plt.draw()
    plt.pause(0.001)   

plt.figure()
plt.plot(cost_collect , 'r')
    
plt.ioff()
plt.show()    