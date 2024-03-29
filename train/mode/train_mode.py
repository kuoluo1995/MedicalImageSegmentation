from train.mode.base import *


class TrainMode(BaseMode):
    name = 'train'

    def set_config(self, **params):
        tf.logging.info('..... setting {} mode config'.format(self.name))
        self.dataset_path = params['dataset_path']

    def get_dataset(self, **params):
        def _dataset_filter(*example_proto):
            with tf.variable_scope('Filter'):
                # todo improve 过滤函数
                return True

        def _data_preprocess(example_proto):
            # 将example转化成数据
            feature, label = example.read_example(example_proto, self, **params)
            with tf.variable_scope('Augmentation'):
                # todo test 数据增广对函数的影响和其他方法
                for function_name in params['image_augmentation']:
                    kwargs = {} if params['image_augmentation'][function_name] == 'None' else \
                        params['image_augmentation'][function_name]
                    feature['image'], new_label = image_process_operations.process_image(function_name=function_name,
                                                                                         image=feature['image'],
                                                                                         label=label, **kwargs,
                                                                                         **params)
                    if new_label is not None:
                        label = new_label

            with tf.variable_scope('UnitedImageSize'):
                image = self.adjust_window_size(feature['image'], params['min_window_level'], params['max_window_level'])
                image = image_process_operations.resize_image(image, params['image_height'],
                                                              params['image_width'])

                feature['image'] = tf.squeeze(image, axis=0)
            with tf.variable_scope('UnitedLabelSize'):
                label = tf.clip_by_value(label, 0, 1)
                label = tf.image.resize_nearest_neighbor(tf.expand_dims(tf.expand_dims(label, axis=0), axis=-1),
                                                         [params['image_height'], params['image_width']])
                label = tf.squeeze(label, axis=(0, -1))
            return feature, label

        with tf.variable_scope('TrainDataset'):
            tf.logging.info('..... get {} dataset'.format(self.name))
            dataset_dict = yaml_tools.read(params['base_path'] / self.dataset_path)
            example = example_tools.create_example(dataset_dict['example'])
            dataset_files = dataset_dict['dataset']
            # 全部文件 数据库处理
            dataset = (Dataset.from_tensor_slices(dataset_files)
                       .shuffle(buffer_size=len(dataset_files), seed=params['seed'])
                       .apply(experimental.parallel_interleave(TFRecordDataset, cycle_length=len(dataset_files))))

            # each_batch 数据库处理
            dataset = (dataset.filter(_dataset_filter)
                       .prefetch(buffer_size=params['batch_size'])
                       .shuffle(buffer_size=dataset_dict['max_fold_size'], seed=params['seed'])
                       .repeat(count=None)
                       .apply(experimental.map_and_batch(map_func=_data_preprocess, batch_size=params['batch_size'],
                                                         num_parallel_batches=params['num_parallel_batches']))
                       .prefetch(buffer_size=contrib_data.AUTOTUNE))
            return dataset

    def init_iterator(self, dataset):
        self.iterator = dataset.make_one_shot_iterator()
        return None

    def adjust_window_size(self, image, min_window, max_window):
        #
        min_window_tensor = tf.random_uniform([], -50, 50)
        max_window_tensor = tf.random_uniform([], -50, 50)
        new_min_window = tf.add(float(min_window), min_window_tensor, name='min_window')
        new_max_window = tf.add(float(max_window), max_window_tensor, name='max_window')
        image = image_process_operations.adjust_window_size(image, new_min_window, new_max_window)
        return image
