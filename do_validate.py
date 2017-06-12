import tensorflow as tf
import numpy as np

def run(sess, net, data_test):
    ph_frame1, ph_frame2, ph_coord = net.placeholders()
    data_test.reset()
    g_loss = net.g_loss
    g_loss_val = 0.0
    while data_test.epoch < 1:
        frame1_batch, frame2_batch, coord_batch = data_test.next_batch()
        frame1_batch = frame1_batch.astype(np.float32) / 127.5 - 1.0
        frame2_batch = frame2_batch.astype(np.float32) / 127.5 - 1.0
        feed_dict = {
            ph_frame1: frame1_batch,
            ph_frame2: frame2_batch,
            ph_coord: coord_batch
        }
        g_loss_val += sess.run(g_loss, feed_dict=feed_dict) * coord_batch.shape[0]
    g_loss_val /= data_test.shape()[0]
    print('Validation: g_loss: %3.6f' % g_loss_val)