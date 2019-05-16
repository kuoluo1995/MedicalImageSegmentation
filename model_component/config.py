import logging
import tensorflow as tf
from pathlib import Path
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging
from tensorflow_estimator.python.estimator.model_fn import ModeKeys


class CustomKeys:
    LEARNING_RATE = "learning_rate"
    METRICS = "metrics"
    GLOBAL_STEP = ops.GraphKeys.GLOBAL_STEP
    SAVERS = ops.GraphKeys.SAVERS
    SUMMARIES = tf.GraphKeys.SUMMARIES
    UPDATE_OPS = tf.GraphKeys.UPDATE_OPS
    TRAIN = ModeKeys.TRAIN
    EVAL = ModeKeys.EVAL
    PREDICT = ModeKeys.PREDICT
    SEPARATOR = "/"
    CLASSES = 'classes'
    PREDICTION = 'prediction'
    LABEL = 'label'


def get_session_config(allow_soft_placement, gpu_options_allow_growth):
    session_config = tf.ConfigProto(allow_soft_placement=allow_soft_placement)
    session_config.gpu_options.allow_growth = gpu_options_allow_growth
    return session_config


def set_logger(tag, mode, log_level=logging.INFO):
    logger = logging.getLogger()
    logger.handlers.clear()  # 每次被调用后，清空已经存在handler
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s: %(levelname).1s %(message)s')
    # 输出到文件
    log_file = Path(__file__).parent.parent / 'logs' / '{}_{}.log'.format(mode, tag)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # 输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    tf_logging._logger = logger
