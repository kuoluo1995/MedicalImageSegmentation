import tensorflow as tf
from abc import abstractmethod
from tensorflow.contrib import slim
from tensorflow.python.ops import init_ops
from train import loss_function
from train import metrics_function
from train.config import CustomKeys


class BaseNet:
    tag = None
    name = None
    init_channels = 64

    feature = None
    image = None
    height = None
    width = None
    batch_size = None
    channel = None

    label = None
    classes = None  # 由于背景类并不是关心的内容。所以就放到代码里写死了

    _logits = None
    _loss = None

    loss_name = None
    train_metrics = None

    def __init__(self):
        tf.logging.info('#### init {} network'.format(self.name))

    @property
    def loss(self):
        return self._loss

    @abstractmethod
    def set_config(self, **params):
        pass

    @abstractmethod
    def _net_args_scope(self, *args, **kwargs):
        pass

    @abstractmethod
    def _build_network(self, image):
        pass

    @abstractmethod
    def _build_prediction(self, logits):
        pass

    @abstractmethod
    def _build_loss(self, logits, label):
        pass

    @abstractmethod
    def _build_metrics(self):
        pass

    @abstractmethod
    def _build_summary(self):
        pass

    def build_model(self, feature, label):
        tf.logging.info('..... building {} model'.format(self.name))
        self.feature = feature  # todo improve 这里未来feature可以传递过来更多有用的信息一起帮助训练
        self.image = self._build_image(feature['image'])
        self.label = self._build_label(label)
        with slim.arg_scope(self._net_args_scope()):
            self._logits = self._build_network(self.image)
            self._build_prediction(self._logits)
        loss = self._build_loss(self._logits, self.label)
        self._build_metrics()
        self._build_summary()
        return loss

    def _build_image(self, image):
        tf.logging.info('................>>>>>>>>>>>>>>>> building image')
        image.set_shape([self.batch_size, self.height, self.width, self.channel])
        return image

    def _build_label(self, label):
        tf.logging.info('................>>>>>>>>>>>>>>>> building label')
        label.set_shape([self.batch_size, self.height, self.width])
        with tf.variable_scope("LabelProcess"):
            one_hot_label = tf.one_hot(label, len(self.classes))
            class_labels = tf.split(one_hot_label, len(self.classes), axis=-1)
            for i, key in enumerate(self.classes):
                self.classes[key]['label'] = class_labels[i]
        return label

    def get_metrics_value(self):
        metrics_dict = dict()
        for key, value in self.classes.items():
            if key == CustomKeys.BACKGROUND:
                continue
            for name, metric in value['metric'].items():
                metrics_dict[key + '_metric' + '_' + name] = metric
        return metrics_dict

    def get_predictions(self):
        # todo improve 这里未来可以加prediction特征值提取
        prediction_dict = dict()
        for key, value in self.classes.items():
            prediction_dict[key + '_label'] = value['label']
            prediction_dict[key + '_prediction'] = value['prediction']
            for name, metric in value['metric'].items():
                prediction_dict[key + '_metric' + '_' + name] = metric
        prediction_dict['name'] = self.feature['name']
        prediction_dict['global_step'] = tf.train.get_or_create_global_step()
        prediction_dict['index'] = self.feature['index']
        return prediction_dict
