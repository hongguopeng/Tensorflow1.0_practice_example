import tensorflow as tf
import numpy as np

def add_layer(inputs , in_size , out_size , n_layer , activaction_function = None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('layer'):  
        with tf.name_scope('weights'):  
            Weights = tf.Variable(tf.random_normal([in_size , out_size]) , name = 'W')
            tf.summary.histogram(layer_name + '/wights' , Weights)	
		       
        with tf.name_scope('biases'):  
            biases = tf.Variable(tf.zeros([1 , out_size]) + 0.1 , name = 'b')
            tf.summary.histogram(layer_name + '/biases' , biases)
			
        with tf.name_scope('Wx_plus_b'):  
            Wx_plus_b = tf.matmul(inputs , Weights) + biases
	
        if activaction_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activaction_function(Wx_plus_b)
            tf.summary.histogram(layer_name + '/outputs' , outputs) 	
			
        return outputs	

# Make up some real data
x_data = np.linspace(-1 , 1 , 300)
x_data = x_data.reshape([x_data.shape[0] , 1])
noise = np.random.normal(0 , 0.05 , x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32 , [None , 1] , name = 'x_input')
    ys = tf.placeholder(tf.float32 , [None , 1] , name = 'y_input')

# add hidden layer
layer_1 = add_layer(xs , 1 , 10 , n_layer = 1 , activaction_function = tf.nn.relu)

# add output layer
prediction = add_layer(layer_1 , 10 , 1 , n_layer = 2 , activaction_function = None)

# the error between prediction and real data
with tf.name_scope('loss'):  
    loss = tf.reduce_mean(
		        tf.reduce_sum(tf.square(ys - prediction) , reduction_indices = 1)
		        )
    tf.summary.scalar('loss' , loss)	# loss會在events下顯示，因此在此不用histogram_summary，改用scalar_summary
#x = np.array([[1 , 1 , 1] , [1 , 1 , 1]])
#tf.reduce_sum(x)     # ->6
#tf.reduce_sum(x, 0)  # ->[2 , 2 , 2]
#tf.reduce_sum(x, 1)  # ->[3 , 3]
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


sess = tf.Session() 
merged = tf.summary.merge_all() # 把所有summary打包在一起
writer = tf.summary.FileWriter('logs/' , sess.graph) # 檔案會存在logs這個資料夾中
sess.run(tf.initialize_all_variables())

for step in range(0 , 1000):
    sess.run(train_step , feed_dict = {xs : x_data , ys : y_data})
    if step % 50 == 0:
        result = sess.run(merged , feed_dict = {xs: x_data , ys : y_data})
        writer.add_summary(result , step)


# 最後在"命令提示" 先指定到logs的上層資料夾
# 在命令提示輸入 : tensorboard --logdir="logs"
# 最後複製網址貼到chrome


