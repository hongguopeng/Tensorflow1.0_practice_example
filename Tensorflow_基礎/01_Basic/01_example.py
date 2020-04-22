import tensorflow as tf
import numpy as np

#---------create data---------#
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3
#---------create data---------#

#---------crete tensorflow structure start---------#
Weights = tf.Variable(tf.random_uniform([1 , 1] , -1.0 , 1.0))
biases = tf.Variable(tf.zeros([1 , 1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 0.5->learning rate
train = optimizer.minimize(loss)

init = tf.initialize_all_variables() # 初始所有神經網路的變數
#---------crete tensorflow structure end---------#


sess = tf.Session() 
sess.run(init)  # 激活神經網路的結構，這一步很容易忘記但也非常重要

for step in range(0 , 201):
    sess.run(train)
    if step % 20 == 0:
        print(step , sess.run(Weights) , sess.run(biases) , sess.run(loss))






























