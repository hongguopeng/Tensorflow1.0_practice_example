import tensorflow as tf

matrix1 = tf.constant([[2 , 2] , [3 , 3]])
matrix2 = tf.constant([[2 , 2] , [3 , 3]])

element_wise = tf.multiply(matrix1 , matrix2) # 矩陣點乘
product = tf.matmul(matrix1 , matrix2) # 矩陣相乘

matrix3 = tf.ones([6 , 1])
matrix4 = tf.random_normal([6 , 10])
matrix5 = tf.concat([matrix3 , matrix4] , axis = 1)

sess = tf.Session()
result1 = sess.run(element_wise)
result2 = sess.run(product)
result3 = sess.run(matrix5)



