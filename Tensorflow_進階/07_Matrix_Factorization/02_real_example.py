import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import trange

"""     ? -> 代表user沒有對item評分
        item1  item2  item3  item4  item5
user1     ?      1      ?      3      ? 

user2     4      ?      ?      2      ?

user3     ?      ?      3      ?      ?

user4     3      ?      1      ?      3

user5     4      3      ?      4      ?

user6     ?      1      ?      4      ?

目的就是要做出一個 m x k 與 k x n 相乘的矩陣可以跟上面的矩陣越像越好(m要大於6 , n要大於5)
而成出來的矩陣的某一個 row 所對應的 column 的數值代表我們預測某一個user對於各個 item 的評分

可以把上述評分表有評分的部分拿出來排成一列
(1 - 1) x 5 + 1 = 1  -> 1 
(1 - 1) x 5 + 3 = 3  -> 3  
(2 - 1) x 5 + 0 = 5  -> 4
(2 - 1) x 5 + 3 = 8  -> 2
(3 - 1) x 5 + 2 = 12 -> 3 
(4 - 1) x 5 + 0 = 15 -> 3  
(4 - 1) x 5 + 2 = 17 -> 1    
(4 - 1) x 5 + 4 = 19 -> 3 
(5 - 1) x 5 + 0 = 20 -> 4
(5 - 1) x 5 + 1 = 21 -> 3
(5 - 1) x 5 + 3 = 23 -> 4         
(6 - 1) x 5 + 1 = 26 -> 1
(6 - 1) x 5 + 3 = 28 -> 4   """ 
  

df = pd.read_csv('user_item_data.data' ,
                 sep = '\t' ,
                 names = ['user' , 'item' , 'rate' , 'time'])
msk = np.random.rand(len(df)) < 0.7
df_train = df.loc[msk]

user_indecies = (np.array(df_train['user']) - 1).reshape([-1 , 1])
item_indecies = (np.array(df_train['item']) - 1).reshape([-1 , 1])
rates = (np.array(df_train['rate'])).reshape([-1 , 1])

# 假設有950個user，一定大於user_indecies.max()
# 假設有1700個item，一定要大於item_indecies.max()
# 否則最後到tf.gather這一步會出錯
user_num , item_num = 950 , 1700
hidden_feature_num = 10
U = tf.Variable(initial_value = tf.truncated_normal([user_num , hidden_feature_num]))
V = tf.Variable(initial_value = tf.truncated_normal([hidden_feature_num , item_num]))

result = tf.matmul(U , V)
result_flatten = tf.reshape(result , [-1])


# 把result_flatten中的 user_indecies * item_num + item_indecies 的項次(上述評分表有評分的部分)挑出來
compute_rates = tf.gather(result_flatten , user_indecies * item_num + item_indecies) 

λ = 0.1
regularization = tf.reduce_mean(U) + tf.reduce_mean(V)
loss = tf.reduce_mean(tf.square(rates - compute_rates)) + λ * regularization
train_step = tf.train.AdamOptimizer(0.02).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost = []
for step in trange(0 , 1000):
    sess.run(train_step)
    if step % 10 == 0:
        cost.append(sess.run(loss))