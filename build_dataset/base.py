import numpy as np
import tensorflow as tf
from abc import abstractmethod
from tensorflow.contrib import data as contrib_data
from tensorflow.python.estimator import util as estimator_util
from tensorflow.python.data import Dataset, experimental, Iterator, TFRecordDataset
from tensorflow.python.framework import dtypes, ops
from tensorflow.python.ops import array_ops
from pathlib import Path

from utils import image_process_operations, example_tools, yaml_tools
from utils.image_tools import ImageTool


def _get_records(records):
    tf_records = []
    for x in records:
        record = Path(__file__).parent.parent / 'data' / x
        tf_records.append(yaml_tools.read(record))
    return tf_records


class BaseDataset:
    name = None
    _source_data_path = None
    _output_data_path = None

    train_scale = None

    k = None
    seed = None
    _k_folds_record = None

    _image_tool = ImageTool(np.int16, is_label=False)
    _label_tool = ImageTool(np.uint8, is_label=True)  # use uint8 to save space
    _examples = None

    batch_size = None
    mode = None
    min_window_level = None
    max_window_level = None
    image_height = None
    image_width = None
    image_channel = 1  # todo improve 改善图像通道的生成方式 一般是灰度，所以先是1
    num_parallel_batches = None

    image_augmentation_params = None

    def __init__(self):
        tf.logging.info('init {} dataset'.format(self.name))

    # *************************************************创建时用的函数************************************************* #
    @abstractmethod
    def set_build_config(self, params):
        pass

    @abstractmethod
    def _get_source_data(self):
        # 获取数据和标签
        image = ''
        label = ''
        return [{image, label}]

    def _get_k_folds(self, source_data):
        def random_split_k_folds(data):
            np.random.seed(self.seed)
            np.random.shuffle(data)

            col_size = len(data) // self.k
            folds = [data[i * col_size:(i + 1) * col_size] for i in range(self.k)]
            remain = data[self.k * col_size:]
            for i, file in enumerate(remain):
                folds[i].append(file)
            return folds

        if not self._k_folds_record.exists():
            dict_ = dict()
            dict_['k'] = self.k
            dict_['k_folds'] = random_split_k_folds(source_data)
            yaml_tools.write(self._k_folds_record, dict_)
        dict_ = yaml_tools.read(self._k_folds_record)
        self.k = dict_['k']
        return dict_['k_folds']

    def save_dataset(self, example_name, dataset, max_fold_size):
        # 保存训练结果
        train_dict = dict()
        train_dict['example'] = example_name
        train_dict['image_channel'] = self.image_channel
        train_dict['max_fold_size'] = max_fold_size
        train_dict['dataset_size'] = int(len(dataset) * self.train_scale)
        train_dict['dataset'] = list(str(dataset[i]) for i in range(train_dict['dataset_size']))
        train_dataset_path = self._output_data_path / 'train_{}.yaml'.format(example_name)
        yaml_tools.write(train_dataset_path, train_dict)

        # 保存测试结果
        eval_dict = dict()
        eval_dict['example'] = example_name
        eval_dict['image_channel'] = self.image_channel
        eval_dict['max_fold_size'] = max_fold_size
        eval_dict['dataset_size'] = len(dataset) - train_dict['dataset_size']
        eval_dict['dataset'] = list(str(dataset[i]) for i in range(train_dict['dataset_size'], len(dataset)))
        eval_dataset_path = self._output_data_path / 'eval_{}.yaml'.format(example_name)
        yaml_tools.write(eval_dataset_path, eval_dict)

    def build_dataset(self):
        # 获取data
        #  - image
        #  - label
        source_data = self._get_source_data()

        # 数据分k份
        k_folds = self._get_k_folds(source_data)

        # 保存成example
        record_fold = self._output_data_path / 'records'
        if not record_fold.exists():
            record_fold.mkdir(parents=True, exist_ok=True)
        for example_name in self._examples:
            dataset = list()
            max_fold_size = 0
            for i, fold in enumerate(k_folds):
                output_path = record_fold / '{}-{}-of-{}.tfrecord'.format(example_name, i + 1, self.k)
                with tf.io.TFRecordWriter(str(output_path)) as example_writer:
                    example = example_tools.create_example(example_name, writer=example_writer,
                                                           image_tool=self._image_tool,
                                                           label_tool=self._label_tool)
                    example.write_example(i, fold)
                dataset.append(output_path)
                max_fold_size = max(max_fold_size, len(fold))
            # 把结果按比例保存成yaml文件
            self.save_dataset(example_name, dataset, max_fold_size)

    # *************************************************训练时用的函数************************************************* #
    @abstractmethod
    def set_train_config(self, **param):
        pass

    def get_data_iterator(self, mode_dict):
        hooks = list()
        feature = None
        with ops.device('/cpu:0'):  # todo test 带测试下。取消这个会如何？
            for key in mode_dict:
                dataset = mode_dict[key].get_dataset(base_path=self._output_data_path, batch_size=self.batch_size,
                                                     min_window_level=self.min_window_level,
                                                     max_window_level=self.max_window_level,
                                                     image_height=self.image_height, image_width=self.image_width,
                                                     seed=self.seed, num_parallel_batches=self.num_parallel_batches,
                                                     image_augmentation=self.image_augmentation_params)

                hook = mode_dict[key].init_iterator(dataset)
                if hook is not None:
                    hooks.append(hook)
                if feature is None:
                    feature = dataset
        # todo improve 合理的拿出feature来的方法
        with tf.name_scope('DatasetIterator/'):
            handler = array_ops.placeholder(dtypes.string, shape=(), name='Handler')
            iterator = Iterator.from_string_handle(handler, feature.output_types, feature.output_shapes,
                                                   feature.output_classes)
            tf.logging.info('build completed dataset iterator')
        return estimator_util.parse_iterator_result(iterator.get_next()), hooks, handler
