import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# make up data
x_data = np.linspace(-7 , 10 , 2500).reshape([-1 , 1])
np.random.shuffle(x_data)
noise = np.random.normal(0 , 8 , x_data.shape)
y_data = np.square(x_data) - 5 + noise


def batch_norm_layer(inputs , on_train):
    # the dimension you wanna normalize, here [0] for batch
    # for image, you wanna do [0 , 1 , 2] for [batch , height , width] but not channel
    fc_mean , fc_var = tf.nn.moments(inputs , axes = [0] , name = 'mean_var')
    
    scale = tf.Variable(tf.ones([1 , inputs.shape[1].value]) , name = 'scale')
    shift = tf.Variable(tf.zeros([1 , inputs.shape[1].value]) , name = 'shift')
    
    ema = tf.train.ExponentialMovingAverage(decay = 0.5)
    ema_apply_op = ema.apply([fc_mean , fc_var])
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(fc_mean) , tf.identity(fc_var)
    
    mean , var = tf.cond(on_train ,    
                         mean_var_with_update ,                               # on_train如果為true的話，會執行這一步，持續更新mean與var
                         lambda:(ema.average(fc_mean) , ema.average(fc_var))) # on_train如果為true的話，會執行這一步，調用上一次的mean與var
                                
    temp = (inputs - mean) / tf.sqrt(var + 0.001) # tf.sqrt(var + 0.001) : 避免分母為0
    outputs = tf.multiply(temp , scale) + shift
  
    return outputs , ema_apply_op
    

def add_layer(inputs , in_size , out_size):
    # weights and biases 
    Weights = tf.Variable(tf.random_normal([in_size , out_size] , mean = 0. , stddev = 1.) , name = 'weight')
    biases = tf.Variable(tf.zeros([1 , out_size]) + 0.1 , name = 'bias')

    # fully connected product
    Wx_plus_b = tf.add(tf.matmul(inputs , Weights) , biases , name = 'output')
    outputs = Wx_plus_b
    
    return outputs
    
xs = tf.placeholder(tf.float32 , [None , 1] , name = 'xs')  # [num_samples , num_features]
ys = tf.placeholder(tf.float32 , [None , 1] , name = 'ys')
on_train = tf.placeholder(tf.bool , name = 'on_train') # train/test selector for batch normalisation

# record input
input_ = []
input_.append(xs)
ema_ = []
           
# build hidden layers & batch normalization layers
N_HIDDEN_UNITS = 30
for l_n in range(0 , 7):
    with tf.variable_scope('layer_' + str(l_n + 1)): 
        L = add_layer(input_[l_n] , input_[l_n].shape[1].value , N_HIDDEN_UNITS)
    
    with tf.variable_scope('layer_norm_' + str(l_n + 1)):
        L_norm , ema = batch_norm_layer(L , on_train)
        #L_norm , ema = batch_norm_layer(L , on_train = tf.constant(True, dtype=tf.bool))
    
    L_act = tf.nn.tanh(L_norm , name = 'layer_act_' + str(l_n + 1))    
    input_.append(L_act)
    ema_.append(ema)    
    
update_ema = tf.group(ema_)    
 
# build output layer
with tf.variable_scope('prediction'):    
    prediction = add_layer(input_[-1] , input_[-1].shape[1].value , 1)
    
#L1 = add_layer(xs , xs.shape[1].value , N_HIDDEN_UNITS)
#L1_norm = batch_norm_layer(L1 , on_train = tf.constant(True, dtype=tf.bool))
#L1_a = tf.nn.tanh(L1_norm)
#L2 = add_layer(L1_a , L1_a.shape[1].value , N_HIDDEN_UNITS)
#L2_norm = batch_norm_layer(L2 , on_train = tf.constant(True, dtype=tf.bool))
#L2_a = tf.nn.tanh(L2_norm)
#L3 = add_layer(L2_a , L2_a.shape[1].value , N_HIDDEN_UNITS)
#L3_norm = batch_norm_layer(L3 , on_train = tf.constant(True, dtype=tf.bool))
#L3_a = tf.nn.tanh(L3_norm)
#L4 = add_layer(L3_a , L3_a.shape[1].value , N_HIDDEN_UNITS)
#L4_norm = batch_norm_layer(L4 , on_train = tf.constant(True, dtype=tf.bool))
#L4_a = tf.nn.tanh(L4_norm)
#L5 = add_layer(L4_a , L4_a.shape[1].value , N_HIDDEN_UNITS)
#L5_norm = batch_norm_layer(L5 , on_train = tf.constant(True, dtype=tf.bool))
#L5_a = tf.nn.tanh(L5_norm)
#L6 = add_layer(L5_a , L5_a.shape[1].value , N_HIDDEN_UNITS)
#L6_norm = batch_norm_layer(L6 , on_train = tf.constant(True, dtype=tf.bool))
#L6_a = tf.nn.tanh(L6_norm)
#prediction = add_layer(L6_a , L6_a.shape[1].value , 1)


cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction) , reduction_indices = [1]) , name = 'cost')
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
tf.add_to_collection('new_way' , train_op) 
# 把train_op放在'new_way'中，下次想要再訓練時也懶得寫太多程式的話，可直接用tf.get_collection('new_way')[0]將train_op調出來
# train_op_retrain = tf.get_collection('new_way')[0]
 
#tf_config = tf.ConfigProto()               #先留著  
#tf_config.gpu_options.allow_growth = True  #先留著
#sess = tf.Session(config = tf_config)      #先留著
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

cost_his = []
record_step = 5
epochs = 100
for k in range(0 , epochs):
    for i in range(0 , len(index)):
        # train on batch
        sess.run([train_op , update_ema] , feed_dict = {xs : x_data[index[i]] , ys : y_data[index[i]] , on_train: True})            
        if i % record_step == 0:
            # record cost
            cost_his.append(sess.run(cost , feed_dict = {xs : x_data , ys : y_data , on_train : True}))


# plot cost curve         
plt.ioff()
plt.figure()
plt.plot(np.arange(len(cost_his)) , np.array(cost_his) , label = 'with BN')     
plt.legend()
plt.show()

# plot prediction curve
plt.ioff()
plt.figure()
x_test = np.linspace(-7 , 10 , 2500).reshape([-1 , 1])
y_test = np.square(x_test) - 5 + np.random.normal(0 , 8 , x_data.shape)
plt.scatter(x_test , y_test)
pred = sess.run(prediction , feed_dict = {xs : x_test , ys : y_test , on_train : False})
plt.plot(x_test , pred , 'r')
plt.show()

# record variable list
train_variable_list = tf.trainable_variables()
all_list = tf.global_variables()
bn_mean_var = []
for g in range(0 , len(all_list)):
    if 'mean_var' in all_list[g].name:
        train_variable_list.append(all_list[g])
        
train_variable_name = []        
for i in range(0 , len(train_variable_list)) : 
    train_variable_name.append(train_variable_list[i].name)         

# save model
saver = tf.train.Saver(var_list = train_variable_list)        
saver.save(sess , 'my_network/save_net')