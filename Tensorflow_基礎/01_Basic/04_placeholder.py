import tensorflow as tf

input1 = tf.placeholder(tf.float32 , [2 , 2]) # [2 , 2]為矩陣維度
input2 = tf.placeholder(tf.float32 , [2 , 2]) # [2 , 2]為矩陣維度

output1 = tf.multiply(input1 , input2) # element-wise的矩陣相乘
output2 = tf.matmul(input1 , input2) # 一般的矩陣相乘

sess = tf.Session() 
print(sess.run(output1 , feed_dict = {input1 : [[7.0 , 7.0] , [7.0 , 7.0]] , input2 : [[3.0 , 3.0] , [3.0 , 3.0]]}))
print(sess.run(output2 , feed_dict = {input1 : [[7.0 , 7.0] , [7.0 , 7.0]] , input2 : [[3.0 , 3.0] , [3.0 , 3.0]]}))



