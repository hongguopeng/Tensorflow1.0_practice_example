#### 請解壓縮data.rar，取得本程式之數據 #####
import tensorflow as tf
import os
import skimage.io
import skimage.transform

sess = tf.Session()
new_saver = tf.train.import_meta_graph(os.path.join('model/transfer_learn.meta'))
new_saver.restore(sess, tf.train.latest_checkpoint(os.path.join('model')))
graph = tf.get_default_graph()

xs_t = graph.get_tensor_by_name('input_layer/xs:0')
ys_t = graph.get_tensor_by_name('input_layer/ys:0')


def load_img(path):
    try:
        img = skimage.io.imread(path)
        img = img / 255.0
        # print 'Original Image Shape: ', img.shape
        # we crop image from center
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
        # resize to 224, 224
        resized_img = skimage.transform.resize(crop_img, (224, 224))[None, :, :, :]   # shape [1, 224, 224, 3]
        return resized_img
    except OSError:
        pass
test_data = load_img(os.path.join('kittycat' , '18662614_d836506830.jpg'))

prediction_t = graph.get_tensor_by_name('fully_connected_layer/out/BiasAdd:0')
prediction = sess.run(prediction_t , feed_dict = {xs_t : test_data})  
