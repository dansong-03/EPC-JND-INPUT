import os
import numpy as np
import tensorflow as tf
import math
import cv2
import MCL_JCI
import MCL_JCI_input

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
DIR = '/home/data'
DATABASE = 'MCL-JCI'
DIS_PATH = DIR + '/' + DATABASE +'/distorted_image'
DIS_JND_PATH = DIR + '/' + DATABASE + '/distorted_image_JND'

# input ,model prediction , and find JND point
def tf_read_image(path):
    image_raw_data_jpg = tf.gfile.FastGFile(path, 'rb').read()
    img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)
    img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8)
    return img_data_jpg

def random_sample(images_x, images_y, patch_size, num_patches):
    """random sample patch pairs from image pairs.
    Args:
        :param images_x, images_y: tensor - (batch_size, height, width, depth).
        :param patch_size: int.
        :param num_patches: int. we crop num_patches patches from each image pair.
    Returns:
        :return patches_x, patches_y: tensor - (batch_size*num_patches, patch_size, patch_size, depth).
    """
    patches_x = []
    patches_y = []

    images_xy = tf.concat([images_x, images_y], axis=2)
    for j in range(num_patches):
        # Randomly crop a [height, width] section of the image.
        patch_xy = tf.random_crop(images_xy[:, :, :], [patch_size, patch_size, 3 * 2])
        patches_x.append(patch_xy[:, :, :3])
        patches_y.append(patch_xy[:, :, 3:])

    patches_x = tf.convert_to_tensor(value=patches_x, dtype=tf.float32, name='sampled_patches_x')
    patches_y = tf.convert_to_tensor(value=patches_y, dtype=tf.float32, name='sampled_patches_y')
    return patches_x, patches_y


def image_convert_patch(comparison, anchor):
    # Input
    comparison_img = tf_read_image(comparison)
    anchor_img = tf_read_image(anchor)
    norm_comparison_img = tf.cast(comparison_img, tf.float32) * (1. / 255) - 0.5
    norm_anchor_img = tf.cast(anchor_img, tf.float32) * (1. / 255) - 0.5
    comparison_patches, anchor_patches = random_sample(norm_comparison_img, norm_anchor_img,
                                                       patch_size=128,
                                                       num_patches=32) # num_patches=64
    return comparison_patches, anchor_patches

def read_img(path):
    return tf.image.decode_image(tf.read_file(path))

# loss or lossless prediction
def lossyOrlossless_predict(comparison1, anchor1, pixel_jnd, compare__jnd):
    # predict  whether compressed image comparison1 is perceptually lossy from reference anchor1 or not
    graph = tf.Graph()
    comparison2 = str(comparison1)
    anchor2 = str(anchor1)
    jnd_map = str(pixel_jnd)
    jnd_compare = str(compare__jnd)
    # print(comparison)
    with graph.as_default() as g:
        keep_prob = tf.placeholder(tf.float32, name='ratio')
        patches_x, patches_y = image_convert_patch(comparison2, anchor2)  # select patches from comparison and anchor
        print("patches_x:", patches_x.shape)
        jnd_x, jnd_y = image_convert_patch(jnd_map, jnd_compare)
        print("jnd_x:", jnd_x.shape)
        # inference model.
        scores = MCL_JCI.inference_jnd(patches_x, patches_y, jnd_x, jnd_y, keep_prob)
        label_pre = tf.cast(tf.round(tf.nn.sigmoid(scores)), tf.int64)  # [1,0,1,1]<0.5 label: 0 and >0.5 label:1
        print("label_pre:", label_pre.shape)
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            MCL_JCI.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

    with tf.Session(graph=graph) as sess:
        checkpoint_file = '/home/project/logs/best_model.ckpt'
        saver.restore(sess, checkpoint_file)
        label = sess.run(label_pre, feed_dict={keep_prob: 1.0})
        print("label:", label.shape)
    return label

if __name__ == '__main__':
    # comparison
    DIR = '/home/data/MCL-JCI/distorted_image'
    ref_img = DIR + '/ImageJND_SRC05/ImageJND_SRC05_100.jpg'
    pixel_jnd = DIS_JND_PATH + '/ImageJND_SRC05/ImageJND_SRC05_100.jpg'
    compare_img = DIR + '/ImageJND_SRC05/ImageJND_SRC05_099.jpg'
    compare__jnd = DIS_JND_PATH + '/ImageJND_SRC05/ImageJND_SRC05_099.jpg'
    pre_label = lossyOrlossless_predict(compare_img, ref_img, pixel_jnd, compare__jnd)
    print('result:', pre_label)







