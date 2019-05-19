import tensorflow as tf
import matplotlib.pyplot as plt  # plt 用于显示图片

slim = tf.contrib.slim
height = 256
width = 256


def print_data(image, label, index, name):
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(0, 1000):
            print("______________________image({})___________________".format(i))
            print_image, print_label, indexed, named = sess.run([tf.squeeze(image), label, index, name])
            if indexed > 95 and indexed < 105:
                plt.imshow(print_image)  # 显示图片
                plt.axis('off')  # 不显示坐标轴
                plt.show()
                plt.imshow(print_label)  # 显示图片
                plt.axis('off')  # 不显示坐标轴
                plt.show()
        coord.request_stop()
        coord.join(threads)


def read_tfrecord(num_samples=26, num_classes=2):
    keys_to_features = {
        'image/image_path': tf.FixedLenFeature([], tf.string),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'segmentation/shape': tf.FixedLenFeature([2], tf.int64),
        'segmentation/encoded': tf.FixedLenFeature([], tf.string),
        'extra/index': tf.FixedLenFeature([], tf.int64)
    }

    items_to_handlers = {
        'image_path': slim.tfexample_decoder.Tensor('image/image_path'),
        'image': slim.tfexample_decoder.Tensor('image/encoded'),
        'image_shape': slim.tfexample_decoder.Tensor('image/shape'),
        'label': slim.tfexample_decoder.Tensor('segmentation/encoded'),
        'label_shape': slim.tfexample_decoder.Tensor('segmentation/shape'),
        'index': slim.tfexample_decoder.Tensor('extra/index')
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    dataset = slim.dataset.Dataset(
        data_sources='E:/PythonProject/ImageSegmentation/dataset/CarcassDataset/records/TwoDExample-2-of-5.tfrecord',
        reader=tf.TFRecordReader, decoder=decoder, num_samples=num_samples, items_to_descriptions=None,
        num_classes=num_classes, )
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset=dataset, num_readers=3, shuffle=True,
                                                              common_queue_capacity=256, common_queue_min=128,
                                                              seed=None)
    name, image, image_shape, label, label_shape, index = provider.get(
        ['image_path', 'image', 'image_shape', 'label', 'label_shape', 'index'])
    image = tf.decode_raw(image, tf.int16)
    image = tf.reshape(image, image_shape)
    image = tf.to_float(image)
    image = tf.image.resize_bilinear(tf.expand_dims(image, axis=0), [height, width])
    image = tf.squeeze(image, axis=0)

    label = tf.decode_raw(label, tf.uint8)
    label = tf.reshape(label, label_shape)
    label = tf.to_int32(label)
    label = tf.clip_by_value(label, 0, 1)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(tf.expand_dims(label, axis=0), axis=-1), [height, width])
    label = tf.squeeze(label, axis=(0, -1))
    label = tf.cast(label * 255, tf.uint8)
    return image, label, index, name


image, label, index, name = read_tfrecord()
print_data(image, label, index, name)
