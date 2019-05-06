import tensorflow as tf

from utils.example_tools import base


class ThreeDExample(base.BaseExample):
    def _create_example_format(self, image_channel):
        self.image_tool.transpose((1, 2, 0))
        self.label_tool.transpose((1, 2, 0))
        image, image_shape = self.image_tool.get_data_and_shape(image_channel=image_channel)
        label, label_shape = self.label_tool.get_data_and_shape(image_channel=image_channel)
        feature = {
            'image/name': base.feature_to_bytes_list(self.image_tool.name),
            'image/shape': base.feature_to_int64_list(image_shape),
            'image/encoded': base.feature_to_bytes_list(image),
            'segmentation/shape': base.feature_to_int64_list(label_shape),
            'segmentation/encoded': base.feature_to_bytes_list(label),
        }
        yield tf.train.Example(features=tf.train.Features(feature=feature))

    def read_example(self, example_proto, mode, **params):
        pass
