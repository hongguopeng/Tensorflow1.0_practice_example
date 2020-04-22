import tensorflow as tf
import os
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2)
X_train , X_val , y_train , y_val = train_test_split(X_train , y_train , test_size = 0.2)


sess = tf.Session()
new_saver = tf.train.import_meta_graph(os.path.join('my_network/save_net.meta'))
new_saver.restore(sess, tf.train.latest_checkpoint(os.path.join('my_network')))


#--------------model parameter saved before--------------#
graph = tf.get_default_graph()
w_a_1_t = graph.get_tensor_by_name('layer_1/w_a:0')
b_a_1_t = graph.get_tensor_by_name('layer_1/b_a:0')
w_a_2_t = graph.get_tensor_by_name('layer_2/w_a:0')
b_a_2_t = graph.get_tensor_by_name('layer_2/b_a:0')
w_o_t = graph.get_tensor_by_name('output_layer/w_o:0')
b_o_t = graph.get_tensor_by_name('output_layer/b_o:0')
w_a_1 = sess.run(w_a_1_t).astype('float32')
b_a_1 = sess.run(b_a_1_t).astype('float32')
w_a_2 = sess.run(w_a_2_t).astype('float32')
b_a_2 = sess.run(b_a_2_t).astype('float32')
w_o = sess.run(w_o_t).astype('float32')
b_o = sess.run(b_o_t).astype('float32')
  
xs_t = graph.get_tensor_by_name('xs:0')
ys_t = graph.get_tensor_by_name('ys:0')
keep_prob_t = graph.get_tensor_by_name('keep_prob:0')
  
middle_1_output_t = graph.get_tensor_by_name('layer_1/output:0')
middle_1_output = sess.run(middle_1_output_t , feed_dict = {xs_t : X_val , keep_prob_t : 1})
middle_2_output_t = graph.get_tensor_by_name('layer_2/output:0')
middle_2_output = sess.run(middle_2_output_t , feed_dict = {xs_t : X_val , keep_prob_t : 1})
prediction_t = graph.get_tensor_by_name('output_layer/output:0')
prediction = sess.run(prediction_t , feed_dict = {xs_t : X_val , keep_prob_t : 1})  

cross_entropy_t = graph.get_tensor_by_name('cross_entropy:0')
cross_entropy = sess.run(cross_entropy_t , feed_dict = {xs_t : X_val , ys_t : y_val , keep_prob_t : 1})
#--------------model parameter saved before--------------#







  