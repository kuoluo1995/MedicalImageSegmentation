import copy
import tensorflow as tf
from abc import abstractmethod
from tensorflow.contrib import slim
from tensorflow.python.ops import init_ops
from model_component import loss_function
from model_component import metrics_function
from model_component.config import CustomKeys


class BaseNet:
    tag = None
    name = None
    init_channels = 64
    is_training = None

    feature = None
    image = None
    batch_size = None

    label = None
    classes = None

    logits = None
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
    def _build_summary(self, label):
        pass

    def build_model(self, feature, label):
        loss = None
        tf.logging.info('..... building {} model'.format(self.name))
        self.feature = feature  # todo improve 这里未来feature可以传递过来更多有用的信息一起帮助训练
        self.image = self._build_image(feature['image'])
        self.label = self._build_label(label)
        with slim.arg_scope(self._net_args_scope()):
            self.logits = self._build_network(self.image)
            self._build_prediction(self.logits)
        if self.is_training:
            loss = self._build_loss(self.logits, label)
        self._build_metrics()
        self._build_summary(label)
        return loss

    def _build_image(self, image):
        tf.logging.info('................>>>>>>>>>>>>>>>> building image')
        image.set_shape([self.batch_size, None, None, None])
        return image

    def _build_label(self, label):
        tf.logging.info('................>>>>>>>>>>>>>>>> building label')
        label.set_shape([self.batch_size, None, None])
        with tf.variable_scope("LabelProcess"):
            one_hot_label = tf.one_hot(label, len(self.classes))
            class_labels = tf.split(one_hot_label, len(self.classes), axis=-1)
            for i, key in enumerate(self.classes):
                self.classes[key][CustomKeys.LABEL] = class_labels[i]
        return label

    def get_predictions(self):
        # todo improve 这里未来可以加prediction特征值提取
        prediction_dict = dict()
        for key, value in self.feature.items():
            prediction_dict[key] = value
        for key, value in self.classes.items():
            if value['show']:
                prediction_dict[
                    CustomKeys.CLASSES + CustomKeys.SEPARATOR + key + CustomKeys.SEPARATOR + CustomKeys.LABEL] = value[
                    CustomKeys.LABEL]
                prediction_dict[
                    CustomKeys.CLASSES + CustomKeys.SEPARATOR + key + CustomKeys.SEPARATOR + CustomKeys.PREDICTION] = \
                    value[CustomKeys.PREDICTION]
                for name, metric in value[CustomKeys.METRICS].items():
                    prediction_dict[
                        CustomKeys.CLASSES + CustomKeys.SEPARATOR + key + CustomKeys.SEPARATOR + CustomKeys.METRICS + CustomKeys.SEPARATOR + name] = metric

        # prediction_dict['index'] = self.feature['index'] todo question 未来加各种特征值检测
        return prediction_dict
