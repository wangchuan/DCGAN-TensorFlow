import tensorflow as tf
import utils as utils

class Net:
    ph_z = None
    ph_image = None

    image_h, image_w = (96, 96)

    L1_lambda = 0.001
    fmap_dim_g = 64
    fmap_dim_d = 64

    batch_size = None

    # batch norm from l2, not l1
    g_bn_l0 = utils.batch_norm(name='g_bn_l0')
    g_bn_l1 = utils.batch_norm(name='g_bn_l1')
    g_bn_l2 = utils.batch_norm(name='g_bn_l2')
    g_bn_l3 = utils.batch_norm(name='g_bn_l3')

    d_bn_l0 = utils.batch_norm(name='d_bn_l0')
    d_bn_l1 = utils.batch_norm(name='d_bn_l1')
    d_bn_l2 = utils.batch_norm(name='d_bn_l2')
    d_bn_l3 = utils.batch_norm(name='d_bn_l3')

    def __init__(self, FLAGS):
        self.batch_size = FLAGS.batch_size
        self.ph_z = tf.placeholder(tf.float32, shape=(self.batch_size, 100), name='z')
        self.ph_image = tf.placeholder(tf.float32, shape=(self.batch_size, self.image_h, self.image_w, 3), name='image')

        self.fake_image = self.generator(self.ph_z)

        self.D, self.D_logits = self.discriminator(self.ph_image, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_image, reuse=True)

        self.g_loss, self.d_loss = self.compute_loss()

        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.d_vars = [var for var in t_vars if 'd_' in var.name]

    def placeholders(self):
        return self.ph_z, self.ph_image

    def generator(self, z):
        s_h, s_w = self.image_h, self.image_w
        s_h2, s_w2 = utils.compute_size(s_h, 2), utils.compute_size(s_w, 2)
        s_h4, s_w4 = utils.compute_size(s_h2, 2), utils.compute_size(s_w2, 2)
        s_h8, s_w8 = utils.compute_size(s_h4, 2), utils.compute_size(s_w4, 2)
        s_h16, s_w16 = utils.compute_size(s_h8, 2), utils.compute_size(s_w8, 2)
        fmap_dim = self.fmap_dim_g
        batch_size = self.batch_size
        with tf.variable_scope("generator") as scope:
            z_ = utils.fc(z, s_h16*s_w16*8*fmap_dim, name='g_l0_fc')
            gl0 = utils.lrelu(self.g_bn_l0(tf.reshape(z_, [batch_size, s_h16, s_w16, fmap_dim*8])))
            gl1 = utils.lrelu(self.g_bn_l1(utils.deconv2d(gl0, [batch_size, s_h8, s_w8, fmap_dim*4], name='g_l1_deconv')))
            gl2 = utils.lrelu(self.g_bn_l2(utils.deconv2d(gl1, [batch_size, s_h4, s_w4, fmap_dim*2], name='g_l2_deconv')))
            gl3 = utils.lrelu(self.g_bn_l3(utils.deconv2d(gl2, [batch_size, s_h2, s_w2, fmap_dim*1], name='g_l3_deconv')))
            gl4 = utils.deconv2d(gl3, [batch_size, s_h, s_w, 3], name='g_l4_deconv')
        return tf.nn.tanh(gl4)

    def discriminator(self, im, reuse):
        fmap_dim = self.fmap_dim_d
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            dl0 = utils.lrelu(utils.conv2d(im, fmap_dim, name='d_l0_conv'))
            dl1 = utils.lrelu(self.d_bn_l0(utils.conv2d(dl0, fmap_dim*2, name='d_l1_conv')))
            dl2 = utils.lrelu(self.d_bn_l1(utils.conv2d(dl1, fmap_dim*4, name='d_l2_conv')))
            dl3 = utils.lrelu(self.d_bn_l2(utils.conv2d(dl2, fmap_dim*8, name='d_l3_conv')))
            dim = 1
            for d in dl3.get_shape()[1:].as_list():
                dim *= d
            dl4 = utils.fc(tf.reshape(dl3, [-1, dim]), 1, name='d_l4_fc')
        return tf.nn.sigmoid(dl4), dl4

    def compute_loss(self):
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))
        d_loss = self.d_loss_real + self.d_loss_fake
        return g_loss, d_loss
