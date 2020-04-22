import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# make up data
x_data = np.linspace(-7 , 10 , 2500).reshape([-1 , 1])
np.random.shuffle(x_data)
noise = np.random.normal(0 , 8 , x_data.shape)
y_data = np.square(x_data) - 5 + noise

class BN_network(object):
    def __init__(self , HIDDEN_UNITS , HIDDEN_LAYER_NUM , LEARNING_RATE , BATCH_NORM):
        self.hidden_units = HIDDEN_UNITS
        self.hidden_layer_num = HIDDEN_LAYER_NUM
        self.learning_rate = LEARNING_RATE
        self.xs = tf.placeholder(tf.float32 , [None , 1] , name = 'xs')  # [num_samples , num_features]
        self.ys = tf.placeholder(tf.float32 , [None , 1] , name = 'ys')
        self.batch_norm = BATCH_NORM
        self.on_train = tf.placeholder(tf.bool , name = 'on_train') # train/test selector for batch normalisation 
        
        with tf.variable_scope('bulid_bn_network'):
            self.bulid_network()
        
        with tf.variable_scope('cost'):
            self.compute_cost()
            
        with tf.variable_scope('train'):
            self.train_model()
            
            
    @staticmethod
    def batch_norm_layer(inputs , on_train):
        # the dimension you wanna normalize, here [0] for batch
        # for image, you wanna do [0 , 1 , 2] for [batch , height , width] but not channel
        fc_mean , fc_var = tf.nn.moments(inputs , axes = [0] , name = 'mean_var')
        
        scale = tf.Variable(tf.ones([1 , inputs.shape[1].value]) , 
                            name = 'scale')
        shift = tf.Variable(tf.zeros([1 , inputs.shape[1].value]) , 
                            name = 'shift')
        
        ema = tf.train.ExponentialMovingAverage(decay = 0.5)
        ema_apply_op = ema.apply([fc_mean , fc_var])
        mean = tf.cond(on_train , lambda : fc_mean , lambda : ema.average(fc_mean))
        var = tf.cond(on_train , lambda : fc_var , lambda : ema.average(fc_var))
                               
        temp = (inputs - mean) / tf.sqrt(var + 0.001) # tf.sqrt(var + 0.001) : 避免分母為0
        outputs = tf.multiply(temp , scale) + shift
      
        return outputs , ema_apply_op
        
    @staticmethod
    def add_layer(inputs , in_size , out_size):
        # weights and biases 
        Weights = tf.Variable(tf.random_normal([in_size , out_size] , mean = 0. , stddev = 1.) , 
                                  name = 'weight')
        biases = tf.Variable(tf.zeros([1 , out_size]) + 0.1 , 
                             name = 'bias')
    
        # fully connected product
        Wx_plus_b = tf.add(tf.matmul(inputs , Weights) , biases , name = 'output')
        outputs = Wx_plus_b
        
        return outputs
    
    def bulid_network(self):    
        # record in
        self.input_ = []
        self.input_.append(self.xs)
        self.ema_ = []
                   
        # build hidden layers & batch normalization layers
        if self.batch_norm:
            for l_n in range(0 , self.hidden_layer_num):
                with tf.variable_scope('layer_' + str(l_n + 1)): 
                    L = self.add_layer(self.input_[l_n] , 
                                       self.input_[l_n].shape[1].value , 
                                       self.hidden_units)
                
                with tf.variable_scope('layer_norm_' + str(l_n + 1)):
                    L_norm , ema = self.batch_norm_layer(L , self.on_train)
                
                L_act = tf.nn.tanh(L_norm , name = 'layer_act_' + str(l_n + 1))    
                self.input_.append(L_act) 
                self.ema_.append(ema)
                
            self.update_ema = tf.group(self.ema_)
        
        else: 
            for l_n in range(0 , self.hidden_layer_num):
                with tf.variable_scope('layer_' + str(l_n + 1)): 
                    L = self.add_layer(self.input_[l_n] , 
                                       self.input_[l_n].shape[1].value , 
                                       self.hidden_units)
                
                L_act = tf.nn.tanh(L , name = 'layer_act_' + str(l_n + 1))    
                self.input_.append(L_act) 
            
         
        # build output layer
        with tf.variable_scope('prediction'):    
            self.prediction = self.add_layer(self.input_[-1] , 
                                             self.input_[-1].shape[1].value , 1)
    def compute_cost(self):    
        self.cost = tf.reduce_mean(tf.reduce_sum(tf.square(self.ys - self.prediction) , reduction_indices = [1]) , name = 'cost')
    
    def train_model(self):
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
            
BN = BN_network(HIDDEN_UNITS = 30 , 
                HIDDEN_LAYER_NUM = 7 , 
                LEARNING_RATE = 0.001 , 
                BATCH_NORM = True)

no_BN = BN_network(HIDDEN_UNITS = 30 , 
                   HIDDEN_LAYER_NUM = 7 , 
                   LEARNING_RATE = 0.001 , 
                   BATCH_NORM = False)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
   
def minibatch_index(minibatch_num , data):
    step = (int(len(data) / minibatch_num) + 1) * minibatch_num
    temp = []
    index = []
    j = 0
    for ii in range(0 , step):
        j = j + 1
        if j > len(data):
            j = j - (len(data))   
        temp.append(j)  
        if len(temp) == minibatch_num:
           index.append(temp)
           temp = []
    index = list(np.array(index) - 1) 
    return index
index = minibatch_index(minibatch_num = 100 , data = x_data)

cost_his_BN = []
record_step = 5
epochs = 100
for k in range(0 , epochs):
    for i in range(0 , len(index)):
        # train on batch
        sess.run([BN.train_op , BN.update_ema] , feed_dict = {BN.xs : x_data[index[i]] , 
                                                              BN.ys : y_data[index[i]] , 
                                                              BN.on_train : True})           
        if i % record_step == 0:
            # record cost
            cost_his_BN.append(sess.run(BN.cost , feed_dict = {BN.xs : x_data , 
                                                               BN.ys : y_data , 
                                                               BN.on_train : True}))                                                                

    
cost_his_no_BN = []
record_step = 5
epochs = 100
for k in range(0 , epochs):
    for i in range(0 , len(index)):
        # train on batch
        sess.run(no_BN.train_op , feed_dict = {no_BN.xs : x_data[index[i]] , 
                                               no_BN.ys : y_data[index[i]]})           
        if i % record_step == 0:
            # record cost
            cost_his_no_BN.append(sess.run(no_BN.cost , feed_dict = {no_BN.xs : x_data , 
                                                                     no_BN.ys : y_data}))         
    
    
    
# plot cost curve         
plt.ioff()
plt.figure()
plt.plot(np.arange(len(cost_his_BN)) , np.array(cost_his_BN) , 'r' , label = 'BN')     
plt.legend()
plt.plot(np.arange(len(cost_his_no_BN)) , np.array(cost_his_no_BN) , 'g' , label = 'no_BN')     
plt.legend()
plt.show()

# plot prediction curve
plt.ioff()
plt.figure()
x_test = np.linspace(-7 , 10 , 2500).reshape([-1 , 1])
y_test = np.square(x_test) - 5 + np.random.normal(0 , 8 , x_data.shape)
plt.scatter(x_test , y_test)

pred_BN = sess.run(BN.prediction , feed_dict = {BN.xs : x_test , 
                                                BN.ys : y_test , 
                                                BN.on_train : False})

pred_no_BN = sess.run(no_BN.prediction , feed_dict = {no_BN.xs : x_test , 
                                                   no_BN.ys : y_test})
     
plt.plot(x_test , pred_BN , 'r' , label = 'BN')
plt.legend()
plt.plot(x_test , pred_no_BN , 'g' , label = 'no_BN')
plt.legend()
plt.show()