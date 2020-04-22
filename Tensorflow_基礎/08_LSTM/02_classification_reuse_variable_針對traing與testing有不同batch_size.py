import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
train_batch_size = 100                   
test_batch_size = 50
n_inputs = 28                       # MNIST data input (img shape: 28*28)
n_steps = 28                        # time steps
n_hidden_units = 128                # neurons in hidden layer
n_classes = 10                      # MNIST classes (0-9 digits)
layer_num = 2

class LSTM(object):
    def __init__(self , batch_size , n_inputs , n_steps , n_hidden_units , n_classes , layer_num , lr):
        self.batch_size = batch_size
        self.n_inputs = n_inputs
        self.n_steps = n_steps
        self.n_hidden_units = n_hidden_units
        self.n_classes = n_classes
        self.layer_num = layer_num
        self.lr = lr
        
        self.xs = tf.placeholder(tf.float32 , [None , n_steps , n_inputs] , name = 'xs')
        self.ys = tf.placeholder(tf.float32 , [None , n_classes] , name = 'ys')

        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
            
        with tf.variable_scope('LSTM_cell'):    
            self.add_cell()
            
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        
        with tf.variable_scope('cost'):
            self.compute_cost()
            
        with tf.variable_scope('train'):    
            self.train_model() 
        
    def add_input_layer(self):
        self.weight_in = tf.get_variable(initializer = tf.random_normal([self.n_inputs , self.n_hidden_units]) , 
                                         name = 'w_in')
        self.bias_in = tf.get_variable(initializer = tf.constant(0.1 , shape = [1 , self.n_hidden_units]) ,
                                       name = 'b_in')
        X = tf.reshape(self.xs , [-1 , self.n_inputs] , name='2_2D')
        X_in_temp = tf.matmul(X , self.weight_in) + self.bias_in
        self.X_in = tf.reshape(X_in_temp , [-1 , self.n_steps , self.n_hidden_units] , name='2_3D')
        
    def add_cell(self):
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden_units, forget_bias = 1.0 , state_is_tuple = True)
        mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.layer_num , state_is_tuple = True) 
        self.init_state = mlstm_cell.zero_state(self.batch_size , dtype = tf.float32)
        
        state = self.init_state
        self.states = []
        self.outputs = []
        with tf.variable_scope('LSTM'):
            for timestep in range(0 , self.n_steps):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                mlstm_cell_output , state = mlstm_cell(self.X_in[: , timestep , :] , state)
                self.outputs.append(mlstm_cell_output) 
                self.states.append(state)
                
    def add_output_layer(self):
        self.weight_out = tf.get_variable(initializer = tf.random_normal([self.n_hidden_units , self.n_classes]) , 
                                          name = 'w_out')
        self.bias_out = tf.get_variable(initializer = tf.constant(0.1 , shape = [1 , self.n_classes]) , 
                                        name = 'b_out')      
        self.pred = tf.add(tf.matmul(self.outputs[-1] , self.weight_out) , self.bias_out , name='2_2D')   
               
    def compute_cost(self):
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.pred , labels = self.ys) , name='average_cost')
        
    def train_model(self):
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
    
        
with tf.variable_scope('RNN_LSTM'):      
    model_train = LSTM(batch_size = train_batch_size , 
                       n_inputs = n_inputs , 
                       n_steps = n_steps , 
                       n_hidden_units = n_hidden_units ,
                       n_classes = n_classes , 
                       layer_num = layer_num ,
                       lr = lr)    
   
# 在 training RNN 和 test RNN 的時候，可能會有不同的 batch_size
# 這將會影響到整個 RNN 的結構，所以導致在 testing 的時候，不能單純地使用 training 時建立的那個 RNN
# 但是 training RNN 和 test RNN 又必須是有同樣的 weights biases 的參數
# 所以此時必須使用 reuse variable 
    tf.get_variable_scope().reuse_variables()
    model_test = LSTM(batch_size = test_batch_size , 
                      n_inputs = n_inputs , 
                      n_steps = n_steps ,   
                      n_hidden_units = n_hidden_units ,
                      n_classes = n_classes , 
                      layer_num = layer_num ,
                      lr = lr) 

train_variable_list = tf.trainable_variables()
train_variable_name = []
for i in train_variable_list:
    train_variable_name.append(i.name)
global_variable_list = tf.global_variables()
global_variable_name = []   
for i in global_variable_list:
    global_variable_name.append(i.name) 
    

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(0 , 1000):   
    batch_xs , batch_ys = mnist.train.next_batch(train_batch_size)
    batch_xs = batch_xs.reshape([train_batch_size , n_steps , n_inputs])
    
    if step == 0:
        feed_dict = {model_train.xs : batch_xs , model_train.ys : batch_ys}
    else:
        feed_dict = {model_train.xs : batch_xs , model_train.ys : batch_ys , model_train.init_state : state_current}
    
    sess.run(model_train.train_op , feed_dict)    
    state_current = sess.run(model_train.states[-1] , feed_dict)
    
    if step % 20 == 0:
        index = np.random.choice(np.arange(test_batch_size) , 
                                 size = 50 , 
                                 replace = False)
        test_xs = batch_xs[index , :].reshape([50 , n_steps , n_inputs])
        test_ys = batch_ys[index , :]
        prediction = sess.run(model_test.pred , feed_dict = {model_test.xs : test_xs , model_test.ys : test_ys})
        prediction = np.argmax(prediction , 1)
        true_label = np.argmax(test_ys , 1)
        accuracy = (prediction == true_label).astype('float32').sum() / test_batch_size
        print(accuracy)

