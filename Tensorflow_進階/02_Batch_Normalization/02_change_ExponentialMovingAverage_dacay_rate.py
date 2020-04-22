import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# make up data
x_data = np.linspace(-7 , 10 , 2500).reshape([-1 , 1])
np.random.shuffle(x_data)
noise = np.random.normal(0 , 8 , x_data.shape)
y_data = np.square(x_data) - 5 + noise


def batch_norm_layer(inputs , on_train , iteration):
    # the dimension you wanna normalize, here [0] for batch
    # for image, you wanna do [0 , 1 , 2] for [batch , height , width] but not channel
    fc_mean , fc_var = tf.nn.moments(inputs , axes = [0] , name = 'mean_var')
    
    scale = tf.Variable(tf.ones([1 , inputs.shape[1].value]) , name = 'scale')
    shift = tf.Variable(tf.zeros([1 , inputs.shape[1].value]) , name = 'shift')
    
    # decay = min(0.9 ,（1 + iteration）/（10 + iteration))
    # 不同的iteration會改變decay的值
    ema = tf.train.ExponentialMovingAverage(0.5 , iteration)
    ema_apply_op = ema.apply([fc_mean , fc_var])
    mean = tf.cond(on_train , lambda : fc_mean , lambda : ema.average(fc_mean))
    var = tf.cond(on_train , lambda : fc_var , lambda : ema.average(fc_var))
     
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
itr = tf.placeholder(tf.int32 , name = 'iteration')

# record input & ema
input_ = []
input_.append(xs)
ema_ = []
           
# build hidden layers & batch normalization layers
N_HIDDEN_UNITS = 30
for l_n in range(0 , 7):
    with tf.variable_scope('layer_' + str(l_n + 1)): 
        L = add_layer(input_[l_n] , input_[l_n].shape[1].value , N_HIDDEN_UNITS)
        #L_norm = batch_norm_layer(L , on_train = tf.constant(True, dtype=tf.bool))
        
    with tf.variable_scope('layer_norm_' + str(l_n + 1)):
        L_norm , ema = batch_norm_layer(L , on_train , itr)
    
    L_act = tf.nn.tanh(L_norm , name = 'layer_act_' + str(l_n + 1))    
    input_.append(L_act)
    ema_.append(ema) 
    
update_ema = tf.group(ema_)

 
# build output layer
with tf.variable_scope('prediction'):    
    prediction = add_layer(input_[-1] , input_[-1].shape[1].value , 1)


cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction) , reduction_indices = [1]) , name = 'cost')
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
tf.add_to_collection('new_way' , train_op) 
# 把train_op放在'new_way'中，下次想要再訓練時也懶得寫太多程式的話，可直接用tf.get_collection('new_way')[0]將train_op調出來

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
mean_var_record = []
record_step = 5
epochs = 100
for k in range(0 , epochs):
    for i in range(0 , len(index)):
        # train on batch
        sess.run([train_op , update_ema] , 
                 feed_dict = {xs : x_data[index[i]] , 
                              ys : y_data[index[i]] , 
                              on_train : True , 
                              itr : len(index) * k + (i + 1)}) 
        
        print((1 + len(index) * k + (i + 1)) / (10 + len(index) * k + (i + 1)))
        if i % record_step == 0:
            # record cost
            cost_his.append(sess.run(cost , 
                                     feed_dict = {xs : x_data , 
                                                  ys : y_data , 
                                                  on_train : True , 
                                                  itr : len(index) * k + (i + 1)}))


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

# save model
saver = tf.train.Saver(var_list = train_variable_list)        
saver.save(sess , 'my_network/save_net')




