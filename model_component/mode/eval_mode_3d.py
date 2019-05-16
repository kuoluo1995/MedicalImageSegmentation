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
                feature, label = example.read_example(example_proto, self, **params)
                with tf.variable_scope('UnitedImageSize'):
                    image = tf.image.resize_bilinear(tf.expand_dims(feature['image'], axis=-1),
                                                     [params['image_height'], params['image_width']])
                    image = self.adjust_window_size(image, params['min_window_level'], params['max_window_level'])
                    # image.set_shape([params['image_height'], params['image_width'], dataset_dict['image_channel']])
                    feature['image'] = image
                with tf.variable_scope('UnitedLabelSize'):
                    label = tf.clip_by_value(label, 0, 1)  # todo improve 之后多类别同时识别考虑修改
                    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, axis=-1),
                                                             [params['image_height'], params['image_width']])
                    label = tf.squeeze(label, axis=-1)
                return feature, label

        def flat_map(feature, label):
            def _map(each_image, each_label, image_path):
                each_feature = {'image': each_image, 'image_path': image_path}
                return each_feature, each_label

            repeat_times = tf.shape(feature['image'], out_type=tf.int64)[0]
            with tf.name_scope('InputPipeline/'):
                shape = tf.concat(([dataset_dict['image_channel'] // 2], tf.shape(feature['image'])[1:]), axis=0)
                padding_image = tf.zeros(shape, dtype=feature['image'].dtype)
                concat_images = tf.concat((padding_image, feature['image'], padding_image), axis=0)
                concat_list = [concat_images[x:-dataset_dict['image_channel'] + x + 1 if x < dataset_dict[
                    'image_channel'] - 1 else None] for x in range(dataset_dict['image_channel'])]
                image_multi_channels = tf.concat(concat_list, axis=-1)
                images = Dataset.from_tensor_slices(image_multi_channels)
                labels = Dataset.from_tensor_slices(label)
                image_path = Dataset.from_tensors(feature['image_path']).repeat(repeat_times)
                return Dataset.zip((images, labels, image_path)).map(_map)

        def data_preprocess_2d(feature, label):
            image = tf.image.resize_bilinear(tf.expand_dims(feature['image'], axis=0),
                                             [params['image_height'], params['image_width']])
            feature['image'] = tf.squeeze(image, axis=0)
            return feature, label

        with tf.variable_scope('EvalDataset'):
            tf.logging.info('..... get {} dataset'.format(self.name))
            dataset_dict = yaml_tools.read(params['base_path'] / self.dataset_path)
            example = example_tools.create_example(dataset_dict['example'])
            # todo improve 找一种一次性拿出来的方法
            dataset = tf.data.TFRecordDataset(dataset_dict['dataset'][0])
            for file_name in dataset_dict['dataset'][1:]:
                dataset = dataset.concatenate(tf.data.TFRecordDataset(file_name))
            dataset = (dataset.map(data_preprocess, num_parallel_calls=2).flat_map(flat_map).apply(
                tf.data.experimental.map_and_batch(data_preprocess_2d, params['batch_size'],
                                                   num_parallel_batches=params['num_parallel_batches'])).prefetch(
                buffer_size=contrib_data.AUTOTUNE))
            return dataset

    def adjust_window_size(self, image, min_window, max_window):
        image = image_process_operations.adjust_window_size(image, min_window, max_window)
        return image
