import tensorflow as tf
import matplotlib.pyplot as plt  # plt 用于显示图片

slim = tf.contrib.slim


def print_data(image, label):
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(26):
            print("______________________image({})___________________".format(i))
            print_image, print_label = sess.run([tf.squeeze(image), label])
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
        'image/name': tf.FixedLenFeature([], tf.string),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'segmentation/shape': tf.FixedLenFeature([2], tf.int64),
        'segmentation/encoded': tf.FixedLenFeature([], tf.string),
        'extra/index': tf.FixedLenFeature([], tf.int64)
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Tensor('image/encoded'),
        'image_shape': slim.tfexample_decoder.Tensor('image/shape'),
        'label': slim.tfexample_decoder.Tensor('segmentation/encoded'),
        'label_shape': slim.tfexample_decoder.Tensor('segmentation/shape'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    dataset = slim.dataset.Dataset(
        data_sources='E:/PythonProject/ImageSegmentation/dataset/BodyDataset/records/TwoDExample-1-of-5.tfrecord',
        reader=tf.TFRecordReader, decoder=decoder, num_samples=num_samples, items_to_descriptions=None,
        num_classes=num_classes, )
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset=dataset, num_readers=3, shuffle=True,
                                                              common_queue_capacity=256, common_queue_min=128,
                                                              seed=None)
    image, image_shape, label, label_shape = provider.get(['image', 'image_shape', 'label', 'label_shape'])
    image = tf.decode_raw(image, tf.int16)
    image = tf.reshape(image, image_shape)
    image = tf.to_float(image)
    label = tf.decode_raw(label, tf.uint8)
    label = tf.reshape(label, label_shape)
    label = tf.to_int32(label)
    return image, label

def test():
    image, label = read_tfrecord()
    print_data(image, label)


