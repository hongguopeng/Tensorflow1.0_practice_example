import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

LR = 0.1
REAL_PARAMS = [1.2 , 2.5]
INIT_PARAMS = [[5 , 4],
               [5 , 1],
               [2 , 4.5]]
INIT_PARAMS = INIT_PARAMS[0] # 

x = np.linspace(-1 , 1 , 200 , dtype = np.float32)   # x data

#Test (1): Visualize a simple linear function with two parameters,
#you can change LR to 1 to see the different pattern in gradient descent.

#def y_fun(a , b , x): return a * x + b
#def tf_y_fun(a , b , x): return a * x + b
def y_fun(a , b , x): return np.sin(b * np.cos(a * x))
def tf_y_fun(a , b , x): return tf.sin(b * tf.cos(a * x))

noise = np.random.randn(200) / 10    
y = y_fun(REAL_PARAMS[0] , REAL_PARAMS[1] , x) + noise  # target

# tensorflow graph
t_variable = [[] , []]
for p in range(0 , 2):
    t_variable[p] = tf.Variable(initial_value = INIT_PARAMS[p] , dtype = tf.float32)

pred = tf_y_fun(t_variable[0] , t_variable[1] , x)
mse = tf.reduce_mean(tf.square(y - pred))
train_op = tf.train.GradientDescentOptimizer(LR).minimize(mse)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
a_list , b_list , cost_list = [] , [] , []
for t in range(1000):
    a_ , b_ , mse_ = sess.run([t_variable[0] , t_variable[1] , mse])
    a_list.append(a_)
    b_list.append(b_)
    cost_list.append(mse_)    # record parameter changes
    result , _ = sess.run([pred , train_op])                          # training


# visualization codes:
print('a = ', a_, 'b = ', b_)
plt.figure(1)
plt.scatter(x , y , c = 'b')    # plot data
plt.plot(x , result , 'r-' , lw=2)   # plot line fitting

# 3D cost figure
fig = plt.figure(2)
ax = Axes3D(fig)
a3D , b3D = np.meshgrid(np.linspace(-2 , 7 , 30) , np.linspace(-2 , 7 , 30))  # parameter space
#cost3D = np.array([np.mean(np.square(y_fun(a_ , b_ , x) - y)) for a_ , b_ in zip(a3D.flatten() , b3D.flatten())]).reshape(a3D.shape)
cost3D = np.zeros(a3D.shape)
for i in range(0 , a3D.shape[0]):
    for j in range(0 , a3D.shape[0]):
        cost3D[i , j] = np.mean(np.square(y_fun(a3D[i , j] , b3D[i , j] , x) - y))
ax.plot_surface(a3D , b3D , cost3D , rstride = 1 , cstride = 1 , cmap = plt.get_cmap('rainbow') , alpha = 0.7)
ax.scatter(a_list[0] , b_list[0] , zs=cost_list[0] , s = 1000 , c = 'r')  # initial parameter place
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.plot(a_list , b_list , zs = cost_list , zdir = 'z', c = 'g' , lw = 3)    # plot 3D gradient descent
plt.show()