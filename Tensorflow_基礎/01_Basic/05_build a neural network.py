import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs , in_size , out_size , activaction_function = None):
    Weights = tf.Variable(tf.random_normal([in_size , out_size]))
    biases = tf.Variable(tf.zeros([1 , out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs , Weights) + biases
	
    if activaction_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activaction_function(Wx_plus_b)
		
    return outputs	

# Make up some real data
x_data = np.linspace(-1 , 1 , 300)
x_data = x_data.reshape([x_data.shape[0] , 1])
noise = np.random.normal(0 , 0.05 , x_data.shape)
y_data = x_data ** 2 - 0.5 + noise

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32 , [None , 1])
ys = tf.placeholder(tf.float32 , [None , 1])

# add hidden layer
layer_1 = add_layer(xs , 1 , 30 , activaction_function = tf.nn.relu)

# add output layer
prediction = add_layer(layer_1 , 30 , 1 , activaction_function = None)

# the error between prediction and real data
loss_temp = tf.reduce_sum(tf.square(ys - prediction) , axis = 1)
loss = tf.reduce_mean(loss_temp)
		
#x = np.array([[1 , 1 , 1] , [1 , 1 , 1]])
#tf.reduce_sum(x)     # ->6
#tf.reduce_sum(x, 0)  # ->[2 , 2 , 2]
#tf.reduce_sum(x, 1)  # ->[3 , 3]
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


init = tf.initialize_all_variables()
sess = tf.Session() 
sess.run(init)
for step in range(0 , 1000):
    sess.run(train_step , feed_dict = {xs : x_data , ys : y_data})
    if step % 50 == 0:
        print(sess.run(loss , feed_dict = {xs: x_data , ys : y_data}))
   


