import tensorflow as tf
from data_provider import *
from ae_model import *

sensor = '0'
root = '../dataset/sensors/'
data_dir = root+'dataset_C'+'0'+'_R01_to_R03_'+sensor+'.pkl'
printer_data,printer_test= read_data_sets(data_dir)
root_best = '../best_DCAE'

learning_rate=1e-5
n_epochs = 1000
batch_size=64

data = tf.placeholder(tf.float32, [None, 1250])

datas = tf.reshape(data, [-1, 1, 1250, 1])
model = BiGAN_model(mode="train")
x=model.Encoder(datas)
data_s=model.Generator(x)
data_hat = tf.reshape(data_s, [-1, 1250])

print('pzq',data.shape,data_hat.shape)
loss=tf.nn.l2_loss(data-data_hat)
opt=tf.train.AdamOptimizer(learning_rate).minimize(loss)

logs_path='../Feature_extraction/logs'

tf.summary.scalar("loss", loss)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

saver=tf.train.Saver()
ckpt_path = '../DCAE_model/model-'+sensor+'/model.ckpt'

config1 = tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)
with tf.Session(config=config1) as sess:
    sess.run(tf.global_variables_initializer())
    # create log writer object
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for epoch in range(n_epochs):
        n_batches = int(len(printer_data.images) / batch_size)
        # Loop over all batches
        for i in range(n_batches):
            batch_x, _ = printer_data.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, summary = sess.run([opt, merged_summary_op], feed_dict={data: batch_x})
            # Compute average loss
        if epoch % 20 == 0:
            c = loss.eval(feed_dict={data: batch_x})
            print('epoch {}, training loss {}'.format(epoch, c))
        if epoch % 100 == 0:
            writer.add_summary(summary)
        if epoch % 100 == 0:
            # write log
            save_path = saver.save(sess, ckpt_path, global_step=epoch)
            print('checkpoint saved in %s' % save_path)

    print('Optimization Finished')
    print('Cost:', loss.eval({data: printer_test.images}))

