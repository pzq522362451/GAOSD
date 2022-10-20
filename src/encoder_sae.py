import tensorflow as tf
import numpy as np
import os
from data_provider import read_data_sets
import pickle
import math

def kl_divergence(p, p_hat):
    return p * tf.log(p) - p * tf.log(p_hat) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat)

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)

tf.set_random_seed(150)
np.random.seed(150)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

learning_rate=1e-5
n_epochs = 1000
batch_size=64
reg_term_lambda = 1e-3
p = 0.1
beta = 3
std1 = math.sqrt(6) / math.sqrt(1250+200+1)

root_model = '../SAE-model'
root_data = '../dataset/sensors/'

faults = ['C0','C2','C3','C4','C5','C6','C7','C8']
sensors = [0,1,2]
for fault in faults:
    path_out = '../results/sae/data_encoded_f_'+fault+'_012.pkl'
    train_encoded = []
    test_encoded = []
    for sensor in sensors:
        best_log_dir = root_model+'/model-'+str(sensor)

        data_dir = root_data+'dataset_'+fault+'_R01_to_R03_' + str(sensor) + '.pkl'
        print(data_dir)
        printer_data,printer_test = read_data_sets(data_dir)

        # Build graph
        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, [None, 1250])
        W1 = tf.Variable(tf.random_normal([1250, 100], stddev=0.03), name='W1')
        b1 = tf.Variable(tf.random_normal([100]), name='b1')
        W2 = tf.Variable(tf.random_normal([100, 1250], stddev=0.03), name='W2')
        b2 = tf.Variable(tf.random_normal([1250]), name='b2')
        linear_layer_one_output = tf.add(tf.matmul(x, W1), b1)
        layer_one_output = tf.nn.sigmoid(linear_layer_one_output)
        linear_layer_two_output = tf.add(tf.matmul(layer_one_output, W2), b2)
        y_ = tf.nn.sigmoid(linear_layer_two_output)
        diff = y_ - x
        p_hat = tf.reduce_mean(tf.clip_by_value(layer_one_output, 1e-10, 1.0), axis=0)
        # p_hat = tf.reduce_mean(layer_one_output,axis=1)
        kl = kl_divergence(p, p_hat)
        cost = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1)) + reg_term_lambda * (
                    tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)) + beta * tf.reduce_sum(kl)
        optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(
            cost)

        # Evaluation
        var_dict = {x.op.name: x for x in
                    tf.contrib.framework.get_variables('encoder/')
                    if 'Adam' not in x.name}
        l_cp = tf.train.latest_checkpoint(best_log_dir)
        print('latest checkpoint:',l_cp)
        tf.contrib.framework.init_from_checkpoint(
            l_cp, var_dict)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_encoded.append(sess.run(layer_one_output,
                                    feed_dict={x: printer_data.images}))
            test_encoded.append(sess.run(layer_one_output,
                                    feed_dict={x: printer_test.images}))
    with open(path_out,'wb') as f:
        pickle.dump((np.hstack(np.array(train_encoded)),
                     np.hstack(np.array(test_encoded))),f,pickle.HIGHEST_PROTOCOL)