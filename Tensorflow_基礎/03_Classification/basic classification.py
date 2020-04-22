import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Make up some real data
mnist = input_data.read_data_sets('MNIST_data' , one_hot = True)

def add_layer(inputs , in_size , out_size , activaction_function = None):
    Weights = tf.Variable(tf.random_normal([in_size , out_size]))
    biases = tf.Variable(tf.zeros([1 , out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs , Weights) + biases
	
    if activaction_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activaction_function(Wx_plus_b)
		
    return outputs	

def compute_accuracy(v_xs , v_ys):
    global prediction
    y_pre = sess.run(prediction , feed_dict = {xs : v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre , 1) , tf.argmax(v_ys , 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float64))
    result = sess.run(accuracy , feed_dict = {xs : v_xs , ys : v_ys})	
    return result	
	

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32 , [None , 28 * 28])
ys = tf.placeholder(tf.float32 , [None , 10])

# add output layer
layer_1 = add_layer(xs , 784 , 300 , activaction_function = tf.nn.sigmoid)
prediction = add_layer(layer_1 , 300 , 10 , activaction_function = tf.nn.softmax)
prediction = tf.log(prediction + 1e-9)


# the error between prediction and real data
cross_entropy_temp = -tf.reduce_sum(ys * prediction , axis = 1)
cross_entropy = tf.reduce_mean(cross_entropy_temp)
train_step = tf.train.GradientDescentOptimizer(0.4).minimize(cross_entropy)


init = tf.initialize_all_variables()
sess = tf.Session() 
sess.run(init)
for step in range(0 , 2000):
    batch_xs , batch_ys = mnist.train.next_batch(100) # 每次從mnist這個大的training set提取100筆資料去訓練，並不是一次拿整個training set去訓練(SGD)
    sess.run(train_step , feed_dict = {xs : batch_xs , ys : batch_ys})
    if step % 50 == 0:
        print(compute_accuracy(mnist.test.images , mnist.test.labels))
   
