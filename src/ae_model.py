
import tensorflow as tf
slim = tf.contrib.slim

def read_DATA():
    # Setup placeholder of real data (DATA)
    with tf.variable_scope('data'):
        data = tf.placeholder(tf.float32, [None, 1250])
        data = tf.reshape(data, [-1, 1, 1250, 1])

    return data


class BiGAN_model(object):

    def __init__(self, mode):
        """Basic setup.

        Args:
          mode: "train" or "generate"
        """
        assert mode in ["train", "generate"]
        self.mode = mode

        # hyper-parameters for model
        self.x_dim = 1250
        self.latent_dim = 100
        self.batch_size = 64
        self.num_samples = 16

        # Global step Tensor.
        self.global_step = None
        print('The mode is %s.' % self.mode)
        print('complete initializing model.')
    def Encoder(self,data,is_training=True):
        with tf.variable_scope('encoder') :
            batch_norm_params = {'decay': 0.9,
                                 'epsilon': 0.001,
                                 'is_training': is_training,
                                 'scope': 'batch_norm'}
            with slim.arg_scope([slim.conv2d],
                                kernel_size=[1, 4],
                                stride=[1, 2],
                                activation_fn=tf.nn.leaky_relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params):
                # data: 1 x 1250 x 1 dim
                self.layer1 = slim.conv2d(inputs=data,
                                          num_outputs=2,
                                          normalizer_fn=None,
                                          stride=[1, 2],
                                          scope='layer1')

                # layer1: 1 x 625 x 2 dim
                self.layer2 = slim.conv2d(inputs=self.layer1,
                                          num_outputs=4,
                                          stride=[1, 2],
                                          scope='layer2')

                # layer2: 1 x 313 x 4 dim
                self.layer3 = slim.conv2d(inputs=self.layer2,
                                          num_outputs=8,
                                          kernel_size=[1, 3],
                                          stride=[1, 2],
                                          padding='valid',
                                          scope='layer3')

                # layer3: 1 x 156 x 8 dim
                self.layer4 = slim.conv2d(inputs=self.layer3,
                                          num_outputs=16,
                                          stride=[1, 2],
                                          scope='layer4')

                # layer4: 1 x 78 x 16 dim
                self.layer5 = slim.conv2d(inputs=self.layer4,
                                          num_outputs=32,
                                          kernel_size=[1, 3],
                                          stride=[1, 2],
                                          padding='valid',
                                          scope='layer5')

                # layer5: 1 x 38 x 32 dim
                self.layer6 = slim.conv2d(inputs=self.layer5,
                                          num_outputs=64,
                                          stride=[1, 2],
                                          scope='layer6')

                # layer6: 1 x 19 x 64 dim
                self.layer7 = slim.conv2d(inputs=self.layer6,
                                          num_outputs=128,
                                          kernel_size=[1, 3],
                                          stride=[1, 2],
                                          padding='valid',
                                          scope='layer7')

                # layer7: 1 x 9 x 128 dim
                self.layer8 = slim.conv2d(inputs=self.layer7,
                                          num_outputs=256,
                                          kernel_size=[1, 3],
                                          stride=[1, 2],
                                          scope='layer8')

                # layer8: 1 x 5 x 256 dim
                self.layer9 = slim.conv2d(inputs=self.layer8,
                                          num_outputs=512,
                                          kernel_size=[1, 2],
                                          stride=[1, 2],
                                          scope='layer9')

                # layer9: 1 x 3 x 512 dim
                self.layer10 = slim.conv2d(inputs=self.layer9,
                                           num_outputs=self.latent_dim,
                                           kernel_size=[1, 3],
                                           normalizer_fn=None,
                                           activation_fn=None,
                                           stride=[1, 2],
                                           padding='valid',
                                           scope='layer10')
                # logits = layer6: 1 x 1 x 100 dim

                z_hat = tf.squeeze(self.layer10, axis=[1, 2])
        print(z_hat.shape)
        return z_hat
    def Generator(self, random_z, is_training=True, reuse=False):
        """Generator setup as Decoder

        Args:
          random_z: A float32 Tensor random vector (latent code)
          is_training: boolean whether training mode or generating mode
          reuse: variables reuse flag

        Returns:
          A float32 scalar Tensor of generated times from random vector
        """
        with tf.variable_scope('Generator') as scope:
            if reuse:
                scope.reuse_variables()

            batch_norm_params = {'decay': 0.9,
                                 'epsilon': 0.001,
                                 'is_training': is_training,
                                 'scope': 'batch_norm'}
            with slim.arg_scope([slim.conv2d_transpose],
                                kernel_size=[1, 4],
                                stride=[1, 2],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params):
                # Use full conv2d_transpose instead of projection and reshape
                # random_z: 100 dim
                self.inputs = tf.reshape(random_z, [-1, 1, 1, self.latent_dim])

                # inputs = 1 x 1 x 100 dim
                self.layer1 = slim.conv2d_transpose(inputs=self.inputs,
                                                    num_outputs=512,
                                                    kernel_size=[1, 2],
                                                    stride=[1, 2],
                                                    padding='valid',
                                                    scope='layer1')


                # # layer1: 1 x 2 x 512 dim
                self.layer2 = slim.conv2d_transpose(inputs=self.layer1,
                                                    num_outputs=256,
                                                    kernel_size=[1, 2],
                                                    stride=[1, 2],
                                                    padding='same',
                                                    scope='layer2')

                # # layer2: 1 x 4 x 256 dim
                self.layer3 = slim.conv2d_transpose(inputs=self.layer2,
                                                    num_outputs=128,
                                                    kernel_size=[1, 3],
                                                    stride=[1, 2],
                                                    padding='valid',
                                                    scope='layer3')

                # # layer3: 1 x 9 x 128 dim
                self.layer4 = slim.conv2d_transpose(inputs=self.layer3,
                                                    num_outputs=64,
                                                    kernel_size=[1, 3],
                                                    stride=[1, 2],
                                                    padding='valid',
                                                    scope='layer4')

                # # layer4: 1 x 19 x 64 dim
                self.layer5 = slim.conv2d_transpose(inputs=self.layer4,
                                                    num_outputs=32,
                                                    kernel_size=[1, 4],
                                                    stride=[1, 2],
                                                    padding='same',
                                                    scope='layer5')

                # # layer5: 1 x 38 x 32 dim
                self.layer6 = slim.conv2d_transpose(inputs=self.layer5,
                                                    num_outputs=16,
                                                    kernel_size=[1, 4],
                                                    stride=[1, 2],
                                                    padding='valid',
                                                    scope='layer6')

                # # layer6: 1 x 78 x 16 dim
                self.layer7 = slim.conv2d_transpose(inputs=self.layer6,
                                                    num_outputs=8,
                                                    kernel_size=[1, 2],
                                                    stride=[1, 2],
                                                    padding='same',
                                                    scope='layer7')

                # # layer7: 1 x 156 x 8 dim
                self.layer8 = slim.conv2d_transpose(inputs=self.layer7,
                                                    num_outputs=4,
                                                    kernel_size=[1, 4],
                                                    stride=[1, 2],
                                                    padding='same',
                                                    scope='layer8')

                # # layer8: 1 x 312 x 4 dim

                self.layer9 = slim.conv2d_transpose(inputs=self.layer8,
                                                    num_outputs=2,
                                                    kernel_size=[1, 3],
                                                    stride=[1, 2],
                                                    padding='valid',
                                                    scope='layer9')

                # # layer9: 1 x 625 x 2 dim
                self.layer10 = slim.conv2d_transpose(inputs=self.layer9,
                                                    num_outputs=1,
                                                    kernel_size=[1, 4],
                                                    stride=[1, 2],
                                                    padding='same',
                                                    scope='layer10')
                # # layer10: 1 x 1250 x 1 dim
                generated_data = self.layer10
                print(generated_data.shape)
                return generated_data



