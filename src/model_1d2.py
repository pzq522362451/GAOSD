import tensorflow as tf
slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
class BiGAN_model(object):
    """Adversarial Feature Learning
    implementation based on http://arxiv.org/abs/1605.09782

    "Adversarial Feature Learning
    Jeff Donahue, Philipp Kr\:ahenb\:uhl, and Trevor Darrell
    """

    def __init__(self, mode):
        """Basic setup.

        Args:
          mode: "train" or "generate"
        """
        assert mode in ["train", "generate"]
        self.mode = mode

        # hyper-parameters for model
        self.x_dim = 1250
        self.z_dim = 100
        self.batch_size = FLAGS.batch_size
        self.num_samples = FLAGS.num_samples

        # Global step Tensor.
        self.global_step = None

        print('The mode is %s.' % self.mode)
        print('complete initializing model.')

    def build_inputs(self):
        """Build random_z.

        Returns:
          A float32 Tensor with [batch_size, 1, 1, z_dim]
        """
        # Setup variable of random vector z
        with tf.variable_scope('random_z'):
            self.random_z = tf.placeholder(tf.float32, [None, self.z_dim])

        return self.random_z

    def read_DATA(self):
        # Setup placeholder of real data (DATA)
        with tf.variable_scope('data'):
            self.data = tf.placeholder(tf.float32, [None, self.x_dim])
            self.data_time = tf.reshape(self.data, [-1, 1, self.x_dim, 1])

            return self.data_time

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
                self.inputs = tf.reshape(random_z, [-1, 1, 1, self.z_dim])

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
                                                    kernel_size=[1, 4],
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
                                                    kernel_size=[1, 4],
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

                return generated_data

    def Encoder(self, data, is_training=True, reuse=False):
        """Encoder setup
        G_{z}(x): \hat{z} = p(x | z)

        Args:
          data: A float32 scalar Tensor of real data
          reuse: reuse flag

        Returns:
          logits: A float32 scalar Tensor
        """
        with tf.variable_scope('Encoder') as scope:
            if reuse:
                scope.reuse_variables()

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
                                          kernel_size=[1, 4],
                                          stride=[1, 2],
                                          scope='layer8')

                # layer8: 1 x 5 x 256 dim
                self.layer9 = slim.conv2d(inputs=self.layer8,
                                          num_outputs=512,
                                          kernel_size=[1, 4],
                                          stride=[1, 2],
                                          scope='layer9')

                # layer9: 1 x 3 x 512 dim
                self.layer10 = slim.conv2d(inputs=self.layer9,
                                          num_outputs=self.z_dim,
                                          kernel_size=[1, 3],
                                          normalizer_fn=None,
                                          activation_fn=None,
                                          stride=[1, 2],
                                          padding='valid',
                                          scope='layer10')
                # logits = layer6: 1 x 1 x 100 dim
                self.z_hat = tf.squeeze(self.layer10, axis=[1, 2])

                return self.z_hat

    def Discriminator(self, data, latent_code, reuse=False):
        """Discriminator setup.

        Args:
          data: A float32 scalar Tensor of real data
          latent_code: A flaot32 scalar Tensor of latent code
          reuse: variables reuse flag

        Returns:
          logits: A float32 scalar Tensor
        """
        with tf.variable_scope('Discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            batch_norm_params = {'decay': 0.9,
                                 'epsilon': 0.001,
                                 'scope': 'batch_norm'}
            with slim.arg_scope([slim.conv2d],
                                kernel_size=[1, 4],
                                stride=[1, 2],
                                activation_fn=tf.nn.leaky_relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params):

                self.latent_code_fc = slim.fully_connected(inputs=latent_code,
                                                           num_outputs=1250 * 2,
                                                           scope='latent_code_fc')
                self.latent_code_reshape = tf.reshape(self.latent_code_fc, [-1, 1, 1250, 2])

                # concatnate input + latent code
                # data: 1 x 1250 x 1 dim
                # latent_code: 1 x 1250 x 2 dim
                self.inputs = tf.concat([data, self.latent_code_reshape], axis=3)
                print('data $ latent_cod',data.shape,self.latent_code_reshape.shape)
                print('inputs', self.inputs.shape)

                self.layer1 = slim.conv2d(inputs=self.inputs,
                                          num_outputs=2,
                                          normalizer_fn=None,
                                          stride=[1, 2],
                                          scope='layer1')

                # layer1: 1 x 810 x 2 dim
                self.layer2 = slim.conv2d(inputs=self.layer1,
                                          num_outputs=4,
                                          stride=[1, 2],
                                          scope='layer2')
                # layer2: 1 x 405 x 4 dim
                self.layer3 = slim.conv2d(inputs=self.layer2,
                                          num_outputs=8,
                                          kernel_size=[1, 3],
                                          stride=[1, 2],
                                          padding='valid',
                                          scope='layer3')
                # layer3: 1 x 202 x 8 dim
                self.layer4 = slim.conv2d(inputs=self.layer3,
                                          num_outputs=16,
                                          stride=[1, 2],
                                          scope='layer4')
                # layer4: 1 x 101 x 16 dim
                self.layer5 = slim.conv2d(inputs=self.layer4,
                                          num_outputs=32,
                                          kernel_size=[1, 3],
                                          stride=[1, 2],
                                          padding='valid',
                                          scope='layer5')
                # layer5: 1 x 50 x 32 dim
                self.layer6 = slim.conv2d(inputs=self.layer5,
                                          num_outputs=64,
                                          stride=[1, 2],
                                          scope='layer6')
                # layer6: 1 x 25 x 64 dim
                self.layer7 = slim.conv2d(inputs=self.layer6,
                                          num_outputs=128,
                                          kernel_size=[1, 3],
                                          stride=[1, 2],
                                          padding='valid',
                                          scope='layer7')
                # layer7: 1 x 12 x 128 dim
                self.layer8 = slim.conv2d(inputs=self.layer7,
                                          num_outputs=256,
                                          kernel_size=[1, 4],
                                          stride=[1, 2],
                                          scope='layer8')
                # layer8: 1 x 6 x 256 dim
                self.layer9 = slim.conv2d(inputs=self.layer8,
                                          num_outputs=512,
                                          kernel_size=[1, 4],
                                          stride=[1, 2],
                                          scope='layer9')
                # layer9: 1 x 3 x 512 dim
                self.layer10 = slim.conv2d(inputs=self.layer9,
                                           num_outputs=1,
                                           kernel_size=[1, 3],
                                           normalizer_fn=None,
                                           activation_fn=None,
                                           stride=[1, 2],
                                           padding='valid',
                                           scope='layer10')
                # layer10: 1 x 1 x 1 dim
                discriminator_logits = tf.squeeze(self.layer10, axis=[2, 3])
                # print('ss',discriminator_logits.shape)
                return discriminator_logits

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        if self.mode == "train":
            self.global_step = tf.Variable(initial_value=0,
                                           name='global_step',
                                           trainable=False,
                                           collections=[tf.GraphKeys.GLOBAL_STEP,
                                                        tf.GraphKeys.GLOBAL_VARIABLES])

            print('complete setup global_step.')


    def GANLoss(self, logits, is_real=True, BIGAN_LOSS=True,scope=None,):
        """Computes standard GAN loss between `logits` and `labels`.

        Args:
          logits: A float32 Tensor of logits.
          is_real: boolean, Treu means `1` labeling, False means `0` labeling.

        Returns:
          A scalar Tensor representing the loss value.
        """
        geo_generator=True
        if is_real:
            labels = tf.ones_like(logits)
        else:
            labels = tf.zeros_like(logits)

        if BIGAN_LOSS:
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,
                                                   logits=logits,
                                                   scope=scope)
        else:

            loss = tf.losses.hinge_loss(labels=labels,
                                                  logits=logits,
                                                  scope=scope)


        return loss

    def build(self):
        """Creates all ops for training or generate."""
        self.setup_global_step()
        if self.mode == "generate":
            pass

        else:
            # generating random vector
            self.random_z = self.build_inputs()
            print(self.random_z)
            # # read dataset
            self.real_data = self.read_DATA()
            print(self.real_data)
            # generating times from Generator() via random vector z
            self.generated_data = self.Generator(self.random_z)

            print(self.generated_data)
            # # inference latent codes from Encoder() via real data
            self.inference_codes = self.Encoder(self.real_data)
            print(self.inference_codes)
            d_means, d_std = tf.nn.moments(self.random_z, axes=[0])
            g_means, g_std = tf.nn.moments(self.inference_codes, axes=[0])
            total_diff = tf.norm(d_means - g_means) + tf.norm(d_std - g_std)

            # discriminating tuple (real data, inference_codes) by Discriminator()
            self.real_logits = self.Discriminator(self.real_data, self.inference_codes)#真实的1620+100
            print(self.real_logits)
            # discriminating tuple (generated_data, random_z) by Discriminator()
            self.fake_logits = self.Discriminator(self.generated_data, self.random_z, reuse=True)
            #resue意味着调用之前的参数，（之前是真实数据输入，现在是假数据输入）
            # losses of real with label "1"
            self.loss_D_real = self.GANLoss(logits=self.real_logits, is_real=True, BIGAN_LOSS=True,scope='loss_D_real')
            # losses of fake with label "0"
            self.loss_D_fake = self.GANLoss(logits=self.fake_logits, is_real=False, BIGAN_LOSS=True, scope='loss_D_fake')
            # losses of Discriminator
            with tf.variable_scope('loss_D'):
                 self.loss_Discriminator = self.loss_D_real + self.loss_D_fake

            # losses of real with label "0" that used to fool the Discriminator
            self.loss_G_fake = self.GANLoss(logits=self.real_logits, BIGAN_LOSS=True,is_real=False, scope='loss_G_fake')
            # losses of fake with label "1" that used to fool the Discriminator
            self.loss_G_real = self.GANLoss(logits=self.fake_logits, BIGAN_LOSS=True,is_real=True, scope='loss_G_real')
            # losses of Discriminator
            with tf.variable_scope('loss_G'):
                 self.loss_Generator = self.loss_G_fake + self.loss_G_real

            # generating times for sample

            self.real_sample_data = self.real_data  # actually feed the data.test.next_batch
            # inference latent codes from Encoder() via real data
            self.sample_inference_codes = self.Encoder(self.real_sample_data, is_training=False, reuse=True)

            self.reconstruction_data = self.Generator(self.sample_inference_codes, is_training=False, reuse=True)
            d_means_n, d_std_n = tf.nn.moments(self.random_z, axes=[0])
            g_means_n, g_std_n = tf.nn.moments(self.sample_inference_codes, axes=[0])
            self.total_diff_n = tf.norm(d_means_n - g_means_n) + tf.norm(d_std_n - g_std_n)

            self.real_sample_data2 = self.real_data  # actually feed the data.test.next_batch
            # inference latent codes from Encoder() via real data
            self.sample_inference_codes2 = self.Encoder(self.real_sample_data2, is_training=False, reuse=True)

            d_means_f, d_std_f = tf.nn.moments(self.random_z, axes=[0])
            g_means_f, g_std_f = tf.nn.moments(self.sample_inference_codes2, axes=[0])
            self.total_diff_f = tf.norm(d_means_f - g_means_f) + tf.norm(d_std_f - g_std_f)
            # Separate variables for each function
            self.D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
            self.G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator') + \
                           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')
            # print('sdads',self.D_vars)
            # write summaries
            # Add loss summaries
            tf.summary.scalar('losses/loss_Discriminator', self.loss_Discriminator)
            tf.summary.scalar('losses/loss_Generator', self.loss_Generator)
            tf.summary.scalar('total difference train', total_diff)
            tf.summary.scalar('total difference normal', self.total_diff_n)
            tf.summary.scalar('total difference fault', self.total_diff_f)

            # Add histogram summaries
            for var in self.D_vars:
                 tf.summary.histogram(var.op.name, var)
            for var in self.G_vars:
                 tf.summary.histogram(var.op.name, var)


        print('complete model build.\n')


