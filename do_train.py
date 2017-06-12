from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import pdb
import cv2

import do_validate

def visualize(images, index):
    images = images.astype(np.uint8)
    N, H, W, C = images.shape
    im = np.zeros([H*N, W, C], np.uint8)
    for i in xrange(N):
        im[i*H:i*H+H] = images[i]
    filename = './result/rst' + str(index) + '.png'
    cv2.imwrite(filename, im)

def run(FLAGS, sess, net, saver, data_train, data_test):
    g_loss, d_loss = net.g_loss, net.d_loss
    d_loss_fake, d_loss_real = net.d_loss_fake, net.d_loss_real

    g_loss_summary = tf.summary.scalar('g_loss', g_loss)
    d_loss_summary = tf.summary.scalar('d_loss', d_loss)
    d_loss_fake_summary = tf.summary.scalar('d_loss_fake', d_loss_fake)
    d_loss_real_summary = tf.summary.scalar('d_loss_real', d_loss_real)
    d_loss_summary_all = tf.summary.merge([d_loss_summary, d_loss_fake_summary, d_loss_real_summary])
    summary_writer = tf.summary.FileWriter(FLAGS.log_path, sess.graph)

    g_vars, d_vars = net.g_vars, net.d_vars
    lr = FLAGS.learning_rate
    beta1 = FLAGS.beta1

    g_optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(d_loss, var_list=d_vars)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    ph_z, ph_image = net.placeholders()
    prev_epoch = data_train.epoch
    while data_train.epoch < FLAGS.epoches:
        image_batch = None
        while (True):
            image_batch = data_train.next_batch()
            if image_batch.shape[0] == FLAGS.batch_size:
                break
        if False:
            im1 = image_batch[0]
            cv2.imshow('im1', im1)
            cv2.waitKey(0)

        image_batch = image_batch.astype(np.float32) / 127.5 - 1.0
        z_batch = np.random.uniform(-1, 1, [image_batch.shape[0], 100]).astype(np.float32)
        feed_dict = {
            ph_z: z_batch,
            ph_image: image_batch,
        }
        _, d_loss_val, g_loss_val, summary_str = sess.run([d_optim, d_loss, g_loss, d_loss_summary_all], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, data_train.iteration)

        _, d_loss_val, g_loss_val, summary_str = sess.run([g_optim, d_loss, g_loss, g_loss_summary], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, data_train.iteration)

        _, d_loss_val, g_loss_val, summary_str = sess.run([g_optim, d_loss, g_loss, g_loss_summary], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, data_train.iteration)

        if data_train.iteration % FLAGS.disp == 0:
            print('Iter[%04d]: d_loss: %3.6f, g_loss: %3.6f' % (data_train.iteration, d_loss_val, g_loss_val))
        if prev_epoch != data_train.epoch:
            print('Epoch[%03d] finished' % data_train.epoch, end=' ')
            fake_images = sess.run(net.fake_image, feed_dict=feed_dict)
            fake_images = (fake_images + 1.0) * 127.5
            visualize(fake_images, data_train.epoch)
            #do_validate.run(sess, net, data_test)
            saver.save(sess, os.path.join(FLAGS.log_path, 'model.ckpt'), data_train.iteration)
        prev_epoch = data_train.epoch

