from train.networks.base import *
from train.config import CustomKeys


class UNet(BaseNet):
    name = 'UNet'
    num_down_samples = None

    def set_config(self, **params):
        tf.logging.info('..... setting {} config'.format(self.name))
        # 必填参数
        self.tag = params['tag']
        self.init_channels = params['init_channels']

        self.height = params['image_height']
        self.width = params['image_width']
        self.batch_size = params['batch_size']
        self.channel = params['image_channel']

        self.classes = params['classes']

        self.loss_name = params['loss_name']
        self.train_metrics = params['train_metrics']
        # 选填参数
        self.num_down_samples = params['num_down_samples']

    def _build_network(self, image):
        tf.logging.info('................>>>>>>>>>>>>>>>> building {} network'.format(self.name))
        with tf.variable_scope('UNet'):
            tensor_out = image
            out_channels = self.init_channels
            # 下采样
            encoder_layers = dict()
            for i in range(self.num_down_samples):
                with tf.variable_scope('Encode{:d}'.format(i + 1)):
                    tensor_out = slim.repeat(tensor_out, 2, slim.conv2d, out_channels, 3)
                    encoder_layers['Encode{:d}'.format(i + 1)] = tensor_out
                    tensor_out = slim.max_pool2d(tensor_out, [2, 2])
                out_channels *= 2

            # 下采样和上采样之间的桥
            tensor_out = slim.repeat(tensor_out, 2, slim.conv2d, out_channels, 3, scope="Encode-Decode-Bridge")

            # 上采样
            for i in reversed(range(self.num_down_samples)):
                out_channels /= 2
                with tf.variable_scope("Decode{:d}".format(i + 1)):
                    tensor_out = slim.conv2d_transpose(tensor_out, tensor_out.get_shape()[-1] // 2, 2, 2)
                    tensor_out = tf.concat((encoder_layers['Encode{:d}'.format(i + 1)], tensor_out), axis=-1)
                    tensor_out = slim.repeat(tensor_out, 2, slim.conv2d, out_channels, 3)

            # 最后一把全连接 转化成输出图片 todo test 这个的效果如何 slim.arg_scope
            with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None, normalizer_params=None):
                logits = slim.conv2d(tensor_out, len(self.classes), 1, scope="AdjustChannels")
                return logits

    def _build_prediction(self, logits):
        tf.logging.info('................>>>>>>>>>>>>>>>> building prediction')
        with tf.variable_scope('Prediction'):
            # 化成概率的情况
            logits_probability = slim.softmax(logits)
            # 切割 一个类一个 logits
            classes_logits = tf.split(logits_probability, len(self.classes), axis=-1)
            zeros = tf.zeros_like(classes_logits[0], dtype=tf.uint8)
            ones = tf.ones_like(zeros, dtype=tf.uint8)
            for i, key in enumerate(self.classes):
                self.classes[key]['prediction'] = tf.where(classes_logits[i] > 0.5, ones, zeros, name=key + '_predict')

    def _build_loss(self, logits, label):
        tf.logging.info('................>>>>>>>>>>>>>>>> building loss')
        with tf.variable_scope('Losses'):
            loss_function.get_loss(self.loss_name, logits, label, self.classes)
            total_loss = tf.losses.get_total_loss()
            tf.losses.add_loss(total_loss)
            return total_loss

    def _build_metrics(self):
        tf.logging.info('................>>>>>>>>>>>>>>>> building metrics')
        with tf.variable_scope('Metrics'):
            for i, key in enumerate(self.classes):
                logits = self.classes[key]['prediction']
                label = self.classes[key]['label']
                self.classes[key]['metric'] = dict()
                for name, args in self.train_metrics.items():
                    result = metrics_function.get_mertrics(name, logits, label, args['eps'])
                    self.classes[key]['metric'][name] = result

    def _build_summary(self):
        tf.logging.info('................>>>>>>>>>>>>>>>> building summary')
        # todo improve 未来考虑多通道 ,并且把summary统一起
        tf.summary.image('{}/Image'.format(self.tag), self.image, max_outputs=1,
                         collections=[CustomKeys.SUMMARIES])
        labels = tf.expand_dims(self.label, axis=-1)
        labels_uint8 = tf.cast(labels * 255 / len(self.classes), tf.uint8)
        tf.summary.image('{}/Label'.format(self.tag), labels_uint8, max_outputs=1,
                         collections=[CustomKeys.SUMMARIES])
        for key, value in self.classes.items():
            tf.summary.image('{}/Prediction/{}'.format(self.tag, key), value['prediction'] * 255, max_outputs=1,
                             collections=[CustomKeys.SUMMARIES])
        for loss in loss_function.get_total_loss():
            tf.summary.scalar('{}/{}'.format(self.tag, loss.op.name), loss, collections=[CustomKeys.SUMMARIES])
        for metrics in metrics_function.get_total_mertrics():
            tf.summary.scalar('{}/{}'.format(self.tag, metrics.op.name), metrics, collections=[CustomKeys.SUMMARIES])
