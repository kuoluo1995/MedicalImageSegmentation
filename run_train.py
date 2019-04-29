import tensorflow as tf
import yaml
from tensorflow.python import ops
from tensorflow.python.eager import context
from train.estimator import MyEstimator
from train import config


def main():
    # 训练数据
    estimate = MyEstimator()
    config_file = open('config/train/train_config.yaml', 'rb')
    estimate_config = yaml.load(config_file)
    config.set_logger(estimate_config['tag'])
    tf.logging.info('input params:{}'.format(estimate_config))

    with context.graph_mode():
        with ops.Graph().as_default():
            estimate.set_config(estimate_config)
            tf.logging.info('start training##########################################################################')
            estimate.train()


if __name__ == "__main__":
    main()
