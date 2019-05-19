from model_component.mode.base import *


class EvalMode3D(BaseMode):
    name = 'eval'

    def set_config(self, **params):
        tf.logging.info('..... setting {} mode config'.format(self.name))
        self.dataset_path = params['dataset_path']

    def get_dataset(self, **params):
        def data_preprocess(example_proto):
            with tf.variable_scope('InputPipeline/'):
                # 将example转化成数据
                feature, label3d, = self.example.read_example(example_proto, self, **params)
                with tf.variable_scope('UnitedImageSize'):
                    image3d = tf.image.resize_bilinear(tf.expand_dims(feature['image'], axis=-1),
                                                       [params['image_height'], params['image_width']])
                    feature['image'] = self.adjust_window_size(image3d, params['min_window_level'],
                                                               params['max_window_level'])
                with tf.variable_scope('UnitedLabelSize'):
                    label3d = tf.image.resize_nearest_neighbor(tf.expand_dims(label3d, axis=-1),
                                                               [params['image_height'], params['image_width']])
                    label3d = tf.squeeze(label3d, axis=-1)
                    label3d = tf.clip_by_value(label3d, 0, 1)  # todo improve 之后多类别同时识别考虑修改
                return feature, label3d

        def flat_map(feature, label3d):
            def _map(each_image, each_label):
                feature['image'] = each_image
                return feature, each_label

            with tf.name_scope('InputPipeline/'):
                shape = tf.concat((tf.shape(feature['image'])[:-1], [self.dataset_dict['image_channel'] // 2]), axis=0)
                padding_image = tf.zeros(shape, dtype=feature['image'].dtype)
                image_multi_channels = tf.concat((padding_image, feature['image'], padding_image), axis=-1)
                images = Dataset.from_tensor_slices(image_multi_channels)
                labels = Dataset.from_tensor_slices(label3d)
                return Dataset.zip((images, labels)).map(_map)

        def data_preprocess_2d(feature, label):
            image = tf.image.resize_bilinear(tf.expand_dims(feature['image'], axis=0),
                                             [params['image_height'], params['image_width']])
            feature['image'] = tf.squeeze(image, axis=0)
            return feature, label

        with tf.variable_scope('EvalDataset'):
            tf.logging.info('..... get {} dataset'.format(self.name))
            # todo improve 找一种一次性拿出来的方法
            dataset = tf.data.TFRecordDataset(self.dataset_dict['dataset'][0])
            for file_name in self.dataset_dict['dataset'][1:]:
                dataset = dataset.concatenate(tf.data.TFRecordDataset(file_name))
            dataset = (dataset.map(data_preprocess, num_parallel_calls=2).flat_map(flat_map).apply(
                tf.data.experimental.map_and_batch(data_preprocess_2d, params['batch_size'],
                                                   num_parallel_batches=params['num_parallel_batches'])).prefetch(
                buffer_size=contrib_data.AUTOTUNE))
            return dataset

    def adjust_window_size(self, image, min_window, max_window):
        image = image_process_operations.adjust_window_size(image, min_window, max_window)
        return image

    def get_extra_feature(self, **params):
        self.dataset_dict = yaml_tools.read(params['base_path'] / self.dataset_path)
        self.example = example_tools.create_example(self.dataset_dict['example'])
        return self.example.extra_feature
