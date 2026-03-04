# -*- coding:utf-8 鈥?-

import tensorflow as tf
import os
import input, network

# def tf_read_image(path):
#     image_raw_data_jpg = tf.gfile.FastGFile(path, 'rb').read()
#     img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)
#     img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8)
#     return img_data_jpg

def tf_read_image(path):
    image_raw = tf.gfile.FastGFile(path, 'rb').read()
    img = tf.image.decode_image(image_raw, channels=3)
    img.set_shape([None, None, 3])
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8)
    return img


def fixed_grid_sample(images_x, images_y, patch_size, grid_rows=8, grid_cols=8):
    """Sample aligned patches on a fixed grid to keep inference deterministic."""
    images_x = tf.expand_dims(images_x, axis=0)
    images_y = tf.expand_dims(images_y, axis=0)

    image_shape = tf.shape(images_x)
    image_height = tf.cast(image_shape[1], tf.float32)
    image_width = tf.cast(image_shape[2], tf.float32)
    half_patch = float(patch_size) / 2.0

    y_centers = tf.linspace(half_patch, image_height - half_patch, grid_rows)
    x_centers = tf.linspace(half_patch, image_width - half_patch, grid_cols)
    grid_y = tf.tile(tf.reshape(y_centers, [grid_rows, 1]), [1, grid_cols])
    grid_x = tf.tile(tf.reshape(x_centers, [1, grid_cols]), [grid_rows, 1])

    y1 = (grid_y - half_patch) / image_height
    x1 = (grid_x - half_patch) / image_width
    y2 = (grid_y + half_patch) / image_height
    x2 = (grid_x + half_patch) / image_width

    boxes = tf.stack([y1, x1, y2, x2], axis=-1)
    boxes = tf.reshape(boxes, [grid_rows * grid_cols, 4], name='grid_boxes')
    box_indices = tf.zeros([grid_rows * grid_cols], dtype=tf.int32)

    patches_x = tf.image.crop_and_resize(
        images_x, boxes, box_indices, [patch_size, patch_size], method='bilinear'
    )
    patches_y = tf.image.crop_and_resize(
        images_y, boxes, box_indices, [patch_size, patch_size], method='bilinear'
    )

    return patches_x, patches_y


def image_convert_patch(comparison, anchor):
    # Input锛?distorted image锛?reference image
    comparison_img = tf_read_image(comparison)
    anchor_img = tf_read_image(anchor)
    norm_comparison_img = tf.cast(comparison_img, tf.float32) * (1. / 255) - 0.5
    norm_anchor_img = tf.cast(anchor_img, tf.float32) * (1. / 255) - 0.5
    comparison_patches, anchor_patches = fixed_grid_sample(
        norm_comparison_img, norm_anchor_img,
        patch_size=32,
        grid_rows=8,
        grid_cols=8
    )
    return comparison_patches, anchor_patches

def lossyOrlossless_predict(comparison1, anchor1, pixel_jnd):
    tf.reset_default_graph()

    comparison2 = str(comparison1)
    anchor2 = str(anchor1)
    jnd_map = str(pixel_jnd)

    keep_prob = tf.placeholder(tf.float32, name='ratio')
    patches_x, patches_y = image_convert_patch(comparison2, anchor2)
    print("patches_x:", patches_x.shape)
    jnd_maps, _ = image_convert_patch(jnd_map, jnd_map)

    scores = network.inference_jnd(patches_x, patches_y, jnd_maps, keep_prob)
    label_pre = tf.cast(tf.round(tf.nn.sigmoid(scores)), tf.int64)
    print("label_pre:", label_pre.shape)

    saver = tf.train.Saver()
    checkpoint_file = './checkpoint/PWJND-model/logs/best_model.ckpt'

    with tf.Session() as sess:
        saver.restore(sess, checkpoint_file)
        label = sess.run(label_pre, feed_dict={keep_prob: 1.0})

    return label
import time
if __name__ == '__main__':
    # anchor_img = 'ImageJND_SRC01_100.jpg'
    # compar_img = 'ImageJND_SRC01_026.jpg'
    # jnd_map    = 'ImageJND_SRC01_jnd_map.jpg'
    anchor_img = 'data/ref/ImageJND_SRC19.bmp'
    compar_img = 'data/jpeg/ImageJND_SRC19_007.jpg'
    jnd_map    = 'data/jnd/ImageJND_SRC19_EPCJND.png'
    start_time = time.time()
    # Keep argument order consistent with lossyOrlossless_predict(comparison, anchor, pixel_jnd):
    # comparison = distorted image, anchor = reference image.
    pre_label = lossyOrlossless_predict(compar_img, anchor_img, jnd_map)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Image JND lossy_or_lossless:", pre_label)




