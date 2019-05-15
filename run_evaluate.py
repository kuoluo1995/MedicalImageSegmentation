import os
import tensorflow as tf
import yaml
from tensorflow.python import ops
from tensorflow.python.eager import context
from evaluators import create_evaluator
from train import config


def main():
    config_file = open('config/evaluate/evaluate_config.yaml', 'rb')
    evaluate_config = yaml.load(config_file)
    config.set_logger(config.CustomKeys.EVAL, evaluate_config['tag'])
    tf.logging.info('input params:{}'.format(evaluate_config))
    os.environ['CUDA_VISIBLE_DEVICES'] = evaluate_config['CUDA_VISIBLE_DEVICES']
    evaluator = create_evaluator(evaluate_config['evaluator']['name'])
    with context.graph_mode():
        with ops.Graph().as_default():
            evaluator.set_eval_config(evaluate_config)
            evaluator.evaluate()


if __name__ == "__main__":
    main()
