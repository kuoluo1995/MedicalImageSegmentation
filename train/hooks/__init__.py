import tensorflow as tf
from tensorflow.python.training.training import CheckpointSaverHook, LoggingTensorHook, NanTensorHook, StopAtStepHook
from train.hooks.iterator_string_handle_hook import IteratorStringHandleHook
from train.hooks.best_checkpoint_saver_hook import BestCheckpointSaverHook
from train.hooks.log_learning_rate_hook import LogLearningRateHook


def get_work_hooks(params, *hooks):
    work_hooks = list()
    for hook in hooks:
        if not hook:
            continue
        if isinstance(hook, list):
            work_hooks += hook
    work_hooks.append(StopAtStepHook(params['num_steps']))
    work_hooks.append(NanTensorHook(params['model'].loss))
    work_hooks.append(LoggingTensorHook(params['tensors'], params['config'].log_step_count_steps))
    work_hooks.append(IteratorStringHandleHook(params['mode_dict']))
    work_hooks.append(
        BestCheckpointSaverHook(params['tag'], params['evaluator'], str(params['model_dir']), params['steps_pre_run']))
    work_hooks.append(
        LogLearningRateHook(params['every_steps'], params['steps_pre_run'], str(params['model_dir']), params['tag']))
    tf.logging.info('build completed work hooks')
    return work_hooks


def get_chief_hooks_tuple(params, *hooks):
    checkpoint_hooks = list()
    for hook in hooks:
        if not hook:
            continue
        if isinstance(hook, list):
            checkpoint_hooks += hook
    checkpoint_hooks.append(CheckpointSaverHook(str(params['model_dir']), save_secs=params['config'].save_checkpoints_secs,
                                                save_steps=params['config'].save_checkpoints_steps,
                                                scaffold=params['model'].scaffold))

    return tuple(checkpoint_hooks)
