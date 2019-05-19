import os
import tensorflow as tf
import yaml
from tensorflow.python import ops
from tensorflow.python.eager import context
from model_component.estimator import MyEstimator
from model_component import config


def main():
    # 训练数据
    config_file = 'carcass_3d_unet'
    estimate = MyEstimator()
    estimate_config = yaml.load(open('config/train/' + config_file + '.yaml', 'rb'))
    estimate_config.setdefault('tag', config_file)
    config.set_logger(config.CustomKeys.TRAIN, estimate_config['tag'])
    tf.logging.info('input params:{}'.format(estimate_config))
    os.environ['CUDA_VISIBLE_DEVICES'] = estimate_config['CUDA_VISIBLE_DEVICES']
    with context.graph_mode():
        with ops.Graph().as_default():
            tf.logging.info('#################### setting config ####################')
            estimate.set_train_config(estimate_config)
            tf.logging.info('####################   end config   ####################')
            tf.logging.info('#################### start training ####################')
            estimate.train()
            tf.logging.info('####################  end training  ####################')


if __name__ == "__main__":
    main()
