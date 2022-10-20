import tensorflow as tf
import numpy as np

import os
from data_provider import read_data_sets
import pickle

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)

tf.set_random_seed(150)
np.random.seed(150)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FLAGS = tf.app.flags.FLAGS

# Training Flags
tf.app.flags.DEFINE_string('train_dir', '../model', '')
tf.app.flags.DEFINE_integer('max_epochs', 1000, '')
tf.app.flags.DEFINE_integer('save_epochs', 100, '')
tf.app.flags.DEFINE_integer('summary_steps', 50, '')
tf.app.flags.DEFINE_integer('print_steps', 5, '')
tf.app.flags.DEFINE_integer('batch_size', 64, '')
tf.app.flags.DEFINE_float('learning_rate_D', 1e-5, '')
tf.app.flags.DEFINE_float('learning_rate_G', 1e-5, '')
tf.app.flags.DEFINE_integer('k', 5, 'the number of step of learing D before learning G')
tf.app.flags.DEFINE_integer('num_samples', 16, '')

root_best = '../BiGAN-model'
root_data = '../dataset/sensors/'
from model_1d2 import BiGAN_model

faults = ['C0','C2','C3','C4','C5','C6','C7','C8']

sensors = [9,10,11]
for fault in faults:
    path_out = '../results/bigan/data_encoded_f_'+fault+'_91011.pkl'
    train_encoded = []
    test_encoded = []
    for sensor in sensors:
        best_log_dir = root_best+'/sensor'+str(sensor)

        data_dir = root_data+'dataset_'+fault+'_R01_to_R03_' + str(sensor) + '.pkl'
        print(data_dir)
        printer_data,printer_test = read_data_sets(data_dir)

        # Build the generator and discriminator.
        tf.reset_default_graph()
        model = BiGAN_model(mode="train")
        model.build()

        # Evaluación del generador
        var_dict = {x.op.name: x for x in
                    tf.contrib.framework.get_variables('Encoder/')
                    if 'Adam' not in x.name}
        l_cp = tf.train.latest_checkpoint(best_log_dir)
        print('latest checkpoint:',l_cp)
        tf.contrib.framework.init_from_checkpoint(
            l_cp, var_dict)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_encoded.append(sess.run(model.sample_inference_codes,
                                    feed_dict={model.data: printer_data.images}))
            test_encoded.append(sess.run(model.sample_inference_codes,
                                    feed_dict={model.data: printer_test.images}))
    with open(path_out,'wb') as f:
        pickle.dump((np.hstack(np.array(train_encoded)),
                     np.hstack(np.array(test_encoded))),f,pickle.HIGHEST_PROTOCOL)