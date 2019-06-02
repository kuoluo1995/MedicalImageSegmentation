import os
import tensorflow as tf
import yaml
from tensorflow.python import ops
from tensorflow.python.eager import context
from model_component.evaluators import create_evaluator
from model_component import config


def main():
    config_file = 'new_carcass_3d_unet'
    evaluate_config = yaml.load(open('config/evaluate/' + config_file + '.yaml', 'rb'))
    evaluate_config.setdefault('tag', config_file)
    config.set_logger(config.CustomKeys.EVAL, evaluate_config['tag'])
    tf.logging.info('input params:{}'.format(evaluate_config))
    os.environ['CUDA_VISIBLE_DEVICES'] = evaluate_config['CUDA_VISIBLE_DEVICES']
    evaluator = create_evaluator(evaluate_config['evaluator']['name'], evaluate_config['evaluator'])
    with context.graph_mode():
        with ops.Graph().as_default() as graph:
            evaluator.set_eval_config(evaluate_config)
            evaluator.evaluate(graph)


if __name__ == "__main__":
    main()
