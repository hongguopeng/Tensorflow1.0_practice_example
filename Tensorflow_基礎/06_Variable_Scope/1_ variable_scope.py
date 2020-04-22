import tensorflow as tf
import numpy as np


#-------------setting neuron network structure-------------#
def add_layer(inputs , in_size , out_size , activaction_function = None): 
    Weights = tf.get_variable(initializer = tf.random_normal([in_size , out_size]) , name = 'w_a')
    biases = tf.get_variable(initializer = tf.zeros([1 , out_size]) + 0.1 , name = 'b_a')
    Wx_plus_b = tf.add(tf.matmul(inputs , Weights) , biases) 
	
	  # here to dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b , keep_prob)
	   
    # here to activaction_function
    if activaction_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activaction_function(Wx_plus_b , name = 'output')
        
    return outputs , Weights

def output_layer(inputs , in_size , out_size , activaction_function = None):
    Weights = tf.get_variable(initializer = tf.random_normal([in_size , out_size]) , name = 'w_o')
    biases = tf.get_variable(initializer = tf.zeros([1 , out_size]) + 0.1 , name = 'b_o')
    Wx_plus_b = tf.add(tf.matmul(inputs , Weights) , biases) 
	
    # here to activaction_function
    if activaction_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activaction_function(Wx_plus_b , name = 'output')

    return outputs , Weights


# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32 , name = 'keep_prob')
xs = tf.placeholder(tf.float32 , [None , 64] , name = 'xs')
ys = tf.placeholder(tf.float32 , [None , 10] , name = 'ys')


# add hidden layer
with tf.variable_scope('layer_1') as scope: 
    layer_1  , weight_1_t = add_layer(xs , 64 , 100 , activaction_function = tf.nn.relu)    
    scope.reuse_variables()    
    layer_1_reuse , weight_1_reuse_t = add_layer(xs , 64 , 100 , activaction_function = tf.nn.relu)

with tf.variable_scope('layer_2') as scope: 
    layer_2 , weight_2_t = add_layer(layer_1 , 100 , 50 , activaction_function = tf.nn.tanh)
    scope.reuse_variables()
    layer_2_reuse , weight_2_reuse_t = add_layer(layer_1_reuse , 100 , 50 , activaction_function = tf.nn.tanh)
    
# add output layer
with tf.variable_scope('output_layer'): 
    prediction , name_pre_t = output_layer(layer_2 , 50 , 10 , activaction_function = tf.nn.softmax)
        
sess = tf.Session()
sess.run(tf.global_variables_initializer())
weight_1 = sess.run(weight_1_t)
weight_1_reuse = sess.run(weight_1_reuse_t)
weight_2 = sess.run(weight_2_t)
weight_2_reuse = sess.run(weight_2_reuse_t)

print(weight_1_t.name)
print(weight_1_reuse_t.name)
print(weight_2_t.name)
print(weight_2_reuse_t.name)

vs = tf.trainable_variables()
trainable_variables = []
for v in vs:
    trainable_variables.append(v.name)