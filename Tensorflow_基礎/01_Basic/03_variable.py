import tensorflow as tf

state = tf.Variable(0 , name = 'counter')
one = tf.constant(1)

new_value = tf.add(state , one)
update = tf.assign(state , new_value)

init = tf.initialize_all_variables() # 有定義Variable的話，一定要加這一句

sess = tf.Session() 
sess.run(init)
for _ in range(0 , 3):
    sess.run(update)
    print(sess.run(state))
   