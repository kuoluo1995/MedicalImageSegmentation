from train.mode.base import *


class EvalMode(BaseMode):
    name = 'eval'

    def set_config(self, **params):
        self.dataset_path = params['dataset_path']
        tf.logging.info('set  completed {} mode config'.format(self.name))

    def get_dataset(self, **params):
        def _data_preprocess(example_proto):
            # 将example转化成数据
            feature, label = example.read_example(example_proto, self, **params)
            with tf.name_scope("UnitedImageSize/"):
                image = image_process_operations.resize_image(feature['image'], params['image_height'],
                                                              params['image_width'])
                image = self.adjust_window_size(image, params['min_window_level'], params['max_window_level'])
                feature['image'] = tf.squeeze(image, axis=0)
            with tf.name_scope("UnitedLabelSize/"):
                label = tf.image.resize_nearest_neighbor(tf.expand_dims(tf.expand_dims(label, axis=0), axis=-1),
                                                         [params['image_height'], params['image_width']])
                label = image_process_operations.adjust_window_size(label, 0, 1)  # todo improve 这里有固定值。
                label = tf.squeeze(label, axis=(0, -1))
            return feature, label

        with tf.name_scope('EvalDataset/'):
            dataset_dict = yaml_tools.read(params['base_path'] / self.dataset_path)
            example = example_tools.create_example(dataset_dict['example'])
            # todo improve 找一种一次性拿出来的方法
            dataset = tf.data.TFRecordDataset(dataset_dict['dataset'][0])
            for file_name in dataset_dict['dataset'][1:]:
                dataset = dataset.concatenate(tf.data.TFRecordDataset(file_name))
            dataset = (dataset.apply(tf.data.experimental.map_and_batch(_data_preprocess, params['batch_size'],
                                                                        num_parallel_batches=params[
                                                                            'num_parallel_batches'],
                                                                        drop_remainder=True))
                       .prefetch(buffer_size=contrib_data.AUTOTUNE))
            tf.logging.info('get {} dataset'.format(self.name))
            return dataset

    def init_iterator(self, dataset):
        self.iterator = dataset.make_initializable_iterator()
        hook = util._DatasetInitializerHook(self.iterator)
        return hook

    def adjust_window_size(self, image, min_window, max_window):
        image = image_process_operations.adjust_window_size(image, min_window, max_window)
        return image
