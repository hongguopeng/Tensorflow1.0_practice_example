import tensorflow as tf
import numpy as np
from tqdm import trange

# create fake data (low-rank matrix X)
A = np.random.randn(100 , 3).astype(np.float32)
B = np.random.randn(3 , 100).astype(np.float32)
X = np.matmul(A, B)

# create tensorflow variables to predict low-rank decomposition
Ahat = tf.Variable(tf.random_normal(A.shape))
Bhat = tf.Variable(tf.random_normal(B.shape))
_X = tf.constant(X)

# on each iteration randomly sample 50 rows and 50 columns of X
rows = tf.random_uniform([20] , maxval = 100, dtype = tf.int32)
cols = tf.random_uniform([30] , maxval = 100, dtype = tf.int32)

# sampled versions of X, Ahat, Bhat
X_smp_temp = tf.transpose(tf.gather(_X , rows))
X_smp = tf.transpose(tf.gather(X_smp_temp , cols))


""" A_smp , B_smp= [] , []
for i in range(0 , rows.shape[0].value):
    A_smp.append(Ahat[rows[i] , :])
A_smp = tf.convert_to_tensor(A_smp) 
for j in range(0 , cols.shape[0].value):
    B_smp.append(Bhat[: , cols[j]])
B_smp = tf.transpose(tf.convert_to_tensor(B_smp)) """

A_smp = tf.gather(Ahat, rows)
B_smp_temp = tf.gather(tf.transpose(Bhat) , cols)
B_smp = tf.transpose(B_smp_temp)

# mean squared residual loss function
resid = X_smp - tf.matmul(A_smp , B_smp)
loss = tf.reduce_mean(tf.square(resid))

# training parameters
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

# train for 1000 iterations
sess = tf.Session()
sess.run(tf.global_variables_initializer())
cost = []
for step in trange(0 , 3000):
    sess.run(train_step)
    cost.append(sess.run(loss))
        
# plot learning curve
import matplotlib.pyplot as plt
plt.plot(cost)
plt.show()