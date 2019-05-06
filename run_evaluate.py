import os
import yaml
from tensorflow.python import ops
from tensorflow.python.eager import context
from train.estimator import MyEstimator


def main():
    estimate = MyEstimator()
    config_file = open('config/evaluate/evaluate_config.yaml', 'rb')
    estimate_config = yaml.load(config_file)
    os.environ['CUDA_VISIBLE_DEVICES'] = estimate_config['CUDA_VISIBLE_DEVICES']
    with context.graph_mode():
        with ops.Graph().as_default():
            estimate.set_eval_config(estimate_config)
            estimate.evaluate()


if __name__ == "__main__":
    main()
