import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/" , one_hot=False)

# Visualize encoder setting
# Parameters
learning_rate = 0.01    # 0.01 this learning rate will be better! Tested
training_epochs = 10
batch_size = 200
display_step = 1
# Network Parameters
n_input = 784  # MNIST data input (img shape: 28 * 28)

# tf Graph input (only pictures)
X = tf.placeholder(tf.float32 , [None , n_input])

# hidden layer settings
n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 10
n_hidden_4 = 2
weights = {'encoder_h1': tf.Variable(tf.truncated_normal([n_input , n_hidden_1])) ,
           'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1 , n_hidden_2])) ,
           'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2 , n_hidden_3])) ,
           'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3 , n_hidden_4])) ,
           'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4 , n_hidden_3])) ,
           'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3 , n_hidden_2])) ,
           'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2 , n_hidden_1])) ,
           'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1 , n_input]))}

biases = {'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])) ,
          'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])) ,
          'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])) ,
          'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])) ,
          'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])) ,
          'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])) ,
          'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])) ,
          'decoder_b4': tf.Variable(tf.random_normal([n_input]))}

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.matmul(x , weights['encoder_h1']) + biases['encoder_b1'])
    
    layer_2 = tf.nn.sigmoid(tf.matmul(layer_1 , weights['encoder_h2']) + biases['encoder_b2'])
    
    layer_3 = tf.nn.sigmoid(tf.matmul(layer_2 , weights['encoder_h3']) + biases['encoder_b3'])
    
    layer_4 = tf.matmul(layer_3 , weights['encoder_h4']) + biases['encoder_b4'] # 不需經過activation function
    
    return layer_4


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.matmul(x, weights['decoder_h1']) + biases['decoder_b1'])
    
    layer_2 = tf.nn.sigmoid(tf.matmul(layer_1, weights['decoder_h2']) + biases['decoder_b2'])
    
    layer_3 = tf.nn.sigmoid(tf.matmul(layer_2, weights['decoder_h3']) + biases['decoder_b3'])
    
    layer_4 = tf.nn.sigmoid(tf.matmul(layer_3, weights['decoder_h4']) + biases['decoder_b4'])
    
    return layer_4


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op # Prediction
y_true = X # Targets (Labels) are the input data.

# Define loss and optimizer, minimize the squared error
# cost = tf.reduce_mean(tf.pow(y_true - y_pred , 2)) 
cost = tf.reduce_mean(tf.square(y_true - y_pred)) 
# tf.pow(y_true - y_pred , 2) == tf.square(y_true - y_pred)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# Launch the graph
sess = tf.Session() 
sess.run(tf.global_variables_initializer())
total_batch = int(mnist.train.num_examples / batch_size)

# Training cycle
for epoch in range(0 , training_epochs):
    # Loop over all batches
    for i in range(0 , total_batch):
        batch_xs , batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
        # Run optimization op (backprop) and cost op (to get loss value)
        sess.run(optimizer , feed_dict={X : batch_xs})
        c = sess.run(cost , feed_dict={X : batch_xs})

    # Display logs per epoch step
    if epoch % display_step == 0:
        print('Epoch:' , '%04d' % (epoch + 1) , 'cost=' , '{:.9f}'.format(c))
print('Optimization Finished!')


encoder_result = sess.run(encoder_op , feed_dict = {X : mnist.test.images})
plt.figure(figsize=(20 , 10))
plt.scatter(encoder_result[: , 0] , encoder_result[: , 1] , c = mnist.test.labels)
plt.colorbar()
plt.show()
