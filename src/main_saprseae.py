import tensorflow as tf
from data_provider import *
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#指定在第0块GPU上跑
import math
import tensorflow as tf
def kl_divergence(p, p_hat):
    return p * tf.log(p) - p * tf.log(p_hat) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat)

# def mosaicW1(images, img_h_w, num_of_imgs, f_name):
#
#     figure, axes = plt.subplots(nrows=num_of_imgs, ncols=num_of_imgs)
#     # if file_name == "weights":
#     #     figure, axes = plt.subplots(nrows=10, ncols=20)
#
#     index = 0
#     for axis in axes.flat:
#         image = axis.imshow(images[index, :].reshape(img_h_w, img_h_w),
#                             cmap=plt.cm.gray, interpolation='nearest')
#         axis.set_frame_on(False)
#         axis.set_axis_off()
#         index += 1
#
#     file=f_name+".png"
#     plt.title(f_name, y=12.00,x=-7.0)
#     plt.savefig(file)
#     print("plotted ", file)
#     plt.close()

sensor = '2'
root = '../dataset/sensors/'
data_dir = root+'dataset_C'+'0'+'_R01_to_R03_'+sensor+'.pkl'
printer_data,printer_test= read_data_sets(data_dir)
root_best = '../best_SAE'

# P_LIST = [0.01, 0.1, 0.5, 0.8]
P_LIST=[0.1]
for p_ in P_LIST:
    learning_rate = 1e-5
    epochs = 1000
    batch_size = 64
    reg_term_lambda = 1e-3
    p = p_
    beta = 3
    std1 = math.sqrt(6) / math.sqrt(1250+200+1)

    x = tf.placeholder(tf.float32, [None, 1250])
    y_ = tf.placeholder(tf.float32, [None, 1250])

    W1 = tf.Variable(tf.random_normal([1250, 100], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random_normal([100]), name='b1')

    W2 = tf.Variable(tf.random_normal([100, 1250], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random_normal([1250]), name='b2')

    linear_layer_one_output = tf.add(tf.matmul(x, W1), b1)
    layer_one_output = tf.nn.sigmoid(linear_layer_one_output)

    linear_layer_two_output = tf.add(tf.matmul(layer_one_output,W2),b2)
    y_ = tf.nn.sigmoid(linear_layer_two_output)

    diff = y_ - x

    p_hat = tf.reduce_mean(tf.clip_by_value(layer_one_output,1e-10,1.0),axis=0)

    #p_hat = tf.reduce_mean(layer_one_output,axis=1)
    kl = kl_divergence(p, p_hat)

    cost= tf.reduce_mean(tf.reduce_sum(diff**2,axis=1)) + reg_term_lambda*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)) + beta*tf.reduce_sum(kl)

    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate ,beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost)

    init_op = tf.global_variables_initializer()

    print("Running for P = ", p)

    logs_path = '../logs'

    tf.summary.scalar("loss", cost)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    ckpt_path = '../SAE_model/model-' + sensor + '/model.ckpt'
    with tf.Session() as sess:
       # initialise the variables
       sess.run(init_op)
       writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

       total_batch = int(len(printer_data.images) / batch_size)
       for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = printer_data.next_batch(batch_size=batch_size)
                #print (batch_y)
                _, c = sess.run([optimiser, cost],
                             feed_dict={x: batch_x})

                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

            if epoch % 20 == 0:
                c = cost.eval(feed_dict={x: batch_x})
                print('epoch {}, training loss {}'.format(epoch, c))

            if epoch % 100 == 0:
                # write log
                save_path = saver.save(sess, ckpt_path, global_step=epoch)
                print('checkpoint saved in %s' % save_path)

       print('Optimization Finished')
       print('Cost:', cost.eval({x: printer_test.images}))

