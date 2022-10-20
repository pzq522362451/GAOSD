import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS
def print_sample_data(sample_data1, sample_data2, max_print=FLAGS.num_samples):
    images1 = sample_data1[:max_print, :]
    images1 = images1.reshape([max_print, 64, 38])
    images1 = images1.swapaxes(0, 1)
    images1 = images1.reshape([64, max_print * 38])

    images2 = sample_data2[:max_print, :]
    images2 = images2.reshape([max_print, 64, 38])
    images2 = images2.swapaxes(0, 1)
    images2 = images2.reshape([64, max_print * 38])

    print_images = np.concatenate((images1, images2), axis=0)

    plt.figure(figsize=(max_print, 2))
    plt.axis('off')
    plt.imshow(print_images, cmap='gray')
    plt.show()