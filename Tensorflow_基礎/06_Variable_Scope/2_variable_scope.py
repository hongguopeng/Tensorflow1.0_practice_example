#可以看看Way_0與Way_1、Way_2的差別

#------------------------Way_0------------------------#
import tensorflow as tf
def test():
    with tf.variable_scope('D_layer1'):
        weights1 = tf.Variable(tf.truncated_normal(shape = [5] , stddev = 0.1) , name = 'weights') 
        name1 = weights1.name
    with tf.variable_scope('D_layer2'):
        weights2 = tf.Variable(tf.truncated_normal(shape = [5] , stddev = 0.1) , name = 'weights') 
        name2 = weights2.name
    return name1 , name2
 
with tf.variable_scope('model') as scope:
    name11 , name12 = test()
    name21 , name22 = test()
print(name11)
print(name12)
print(name21)
print(name22)
#------------------------Way_0------------------------#



#------------------------Way_1------------------------#
import tensorflow as tf
def test():
    with tf.variable_scope('D_layer1'):
        weights1 = tf.get_variable('weights', [5] , initializer = tf.truncated_normal_initializer(stddev = 0.1)) 
        name1 = weights1.name
    with tf.variable_scope('D_layer2'):
        weights2 = tf.get_variable('weights', [5] , initializer = tf.truncated_normal_initializer(stddev = 0.1)) 
        name2 = weights2.name
    return name1 , name2
 
with tf.variable_scope('model') as scope:
    name11 , name12 = test()
    scope.reuse_variables()
    name21 , name22 = test()
print(name11)
print(name12)
print(name21)
print(name22)
#------------------------Way_1------------------------#


#------------------------Way_2(補充用法)------------------------#
import tensorflow as tf
def test(reuse = False):
    #在layer1命名空間內創建變量，默認reuse=False
    with tf.variable_scope('D_layer1' , reuse):
        weights1 = tf.get_variable('weights', [5], initializer = tf.truncated_normal_initializer(stddev = 0.1)) 
        name1 = weights1.name
    #在layer2命名空間內創建變量，默認reuse=False
    with tf.variable_scope('D_layer2' ,  reuse):
        weights2 = tf.get_variable('weights', [5], initializer = tf.truncated_normal_initializer(stddev = 0.1)) 
        name2 = weights2.name
    return name1 , name2
 
with tf.variable_scope('model_1') as scope:
    name11 , name12 = test()
with tf.variable_scope('model_2') as scope:
    name21 , name22 = test(reuse = True)
print(name11)
print(name12)
print(name21)
print(name22)
#------------------------Way_2(補充用法)------------------------#

