from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np

from data_provider import read_data_sets
import tensorflow as tf
from model_1d2 import *

slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)

# set random seeds
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

# laod path
sensor = '0'
root = '../dataset/sensors/'

# load training data
data_dir = root+'dataset_C'+'0'+'_R01_to_R03_'+sensor+'.pkl'
printer_data,printer_test= read_data_sets(data_dir)
# load testing data
data_dir = root+'dataset_C'+'2'+'_R01_to_R03_'+sensor+'.pkl'
printer_data_f,printer_test_f = read_data_sets(data_dir)
num_examples = len(printer_data.images)

print(sensor)
root_best = '../best'

model = BiGAN_model(mode="train")
model.build()

#show info for trainable variables
t_vars = tf.trainable_variables()
slim.model_analyzer.analyze_vars(t_vars, print_info=True)

opt_D = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_D, beta1=0.9,beta2=0.999, epsilon=1e-08)
opt_G = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_G, beta1=0.9,beta2=0.999, epsilon=1e-08)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator')):
    opt_D_op = opt_D.minimize(model.loss_Discriminator, var_list=model.D_vars)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator') + \
                             tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Encoder')):
    opt_G_op = opt_G.minimize(model.loss_Generator, global_step=model.global_step,
                              var_list=model.G_vars)


saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

summary_op = tf.summary.merge_all()

sv = tf.train.Supervisor(logdir=FLAGS.train_dir,
                         summary_op=None,
                         saver=saver,
                         save_model_secs=0,
                         init_fn=None)


sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
with sv.managed_session(config=sess_config) as sess:
    tf.logging.info('Start Session.')

    # sv.start_queue_runners(sess=sess)
    # tf.logging.info('Starting Queues.')

    num_batches_per_epoch = num_examples / FLAGS.batch_size

    # save loss values for plot
    losses = []
    best_diff = 0
    for epoch in range(FLAGS.max_epochs + 1):
        for j in range(int(num_batches_per_epoch)):
            start_time = time.time()
            if sv.should_stop():
                break

            for _ in range(FLAGS.k):
                printer_data_batch = printer_data.next_batch(FLAGS.batch_size)
                random_z = np.random.uniform(low=-1., high=1., size=[FLAGS.batch_size, model.z_dim])

                _, loss_D = sess.run([opt_D_op, model.loss_Discriminator],
                                     feed_dict={model.data: printer_data_batch[0],
                                                model.random_z: random_z})
            _, _global_step, loss_G = sess.run([opt_G_op,
                                                sv.global_step,
                                                model.loss_Generator],
                                               feed_dict={model.data: printer_data_batch[0],
                                                          model.random_z: random_z})

            epochs = epoch + j / num_batches_per_epoch
            duration = time.time() - start_time

            if _global_step % FLAGS.print_steps == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                print("Epochs: %.3f global step: %d  loss_D: %g  loss_G: %g (%.1f examples/sec; %.3f sec/batch)"
                      % (epochs, _global_step, loss_D, loss_G, examples_per_sec, duration))

                losses.append([epochs, loss_D, loss_G])

                # print sample data
                sample_printer_batch_f = printer_test_f.next_batch(FLAGS.batch_size)
                diff_f = sess.run(model.total_diff_f,
                                                          feed_dict={model.data: sample_printer_batch_f[0],
                                                                     model.random_z: random_z})
                print('diff f:',diff_f)

                sample_printer_batch = printer_test.next_batch(FLAGS.batch_size)
                diff_n = sess.run(model.total_diff_n,
                                  feed_dict={model.data: sample_printer_batch[0],
                                             model.random_z: random_z})
                print('diff n:', diff_n)
                if diff_f-diff_n > best_diff:
                    print('lower:',diff_n,'gap:',diff_f-diff_n)
                    best_diff = diff_f-diff_n
                    # save model checkpoint periodically
                    tf.logging.info(
                        'Saving best model to disk.')
                    sv.saver.save(sess, root_best+"/sensor"+sensor+"/model_best", global_step=sv.global_step)

            # write summaries periodically
            if _global_step % FLAGS.summary_steps == 0:
                summary_str = sess.run(summary_op, feed_dict={model.data: printer_data_batch[0],
                                                              model.random_z: random_z})
                sv.summary_computed(sess, summary_str)

            # save model checkpoint periodically
            if epoch != 0 and epoch % FLAGS.save_epochs == 0 and j == 0:
                tf.logging.info('Saving model with global step %d (= %d epochs) to disk.' % (_global_step, epoch))
                sv.saver.save(sess, sv.save_path, global_step=sv.global_step)
            if epoch == FLAGS.max_epochs:
                break

    tf.logging.info('complete training...')
