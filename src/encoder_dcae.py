import tensorflow as tf
import numpy as np
import os
from data_provider import read_data_sets
import pickle
from ae_model import *

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)

tf.set_random_seed(150)
np.random.seed(150)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

root_best = '../DCAE-model'
root_data = '../dataset/sensors/'

learning_rate=1e-5
n_epochs = 1000
batch_size=64

faults = ['C0','C2','C3','C4','C5','C6','C7','C8']
# faults = ['C0']
sensors = [9,10,11]
for fault in faults:
    path_out = '../results/dcae/data_encoded_f_'+fault+'_91011.pkl'
    train_encoded = []
    test_encoded = []
    for sensor in sensors:
        best_log_dir = root_best+'/model-'+str(sensor)

        data_dir = root_data+'dataset_'+fault+'_R01_to_R03_' + str(sensor) + '.pkl'
        print(data_dir)
        printer_data,printer_test = read_data_sets(data_dir)

        # Build the generator and discriminator.
        tf.reset_default_graph()
        data = tf.placeholder(tf.float32, [None, 1250])
        datas = tf.reshape(data, [-1, 1, 1250, 1])
        model = BiGAN_model(mode="train")
        x = model.Encoder(datas)
        data_s = model.Generator(x)
        data_hat = tf.reshape(data_s, [-1, 1250])
        loss = tf.nn.l2_loss(data - data_hat)
        opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # Evaluación del generador
        var_dict = {x.op.name: x for x in
                    tf.contrib.framework.get_variables('encoder/')
                    if 'Adam' not in x.name}
        l_cp = tf.train.latest_checkpoint(best_log_dir)
        print('latest checkpoint:',l_cp)
        tf.contrib.framework.init_from_checkpoint(
            l_cp, var_dict)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_encoded.append(sess.run(x,
                                    feed_dict={data: printer_data.images}))
            test_encoded.append(sess.run(x,
                                    feed_dict={data: printer_test.images}))
    with open(path_out,'wb') as f:
        pickle.dump((np.hstack(np.array(train_encoded)),
                     np.hstack(np.array(test_encoded))),f,pickle.HIGHEST_PROTOCOL)