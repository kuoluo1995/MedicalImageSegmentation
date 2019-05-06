import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.python.framework import ops
from tensorflow.python.training import training_util, saver as saver_lib
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.python.training.summary_io import SummaryWriterCache
from train.config import CustomKeys
from utils import summary_tools

from utils import yaml_tools


class BestCheckpointSaverHook(SessionRunHook):
    def __init__(self, tag, evaluator, checkpoint_dir, steps_pre_run, checkpoint_name='best_model.ckpt'):
        self.evaluator = evaluator
        self.checkpoint_dir = checkpoint_dir

        self._summary_tag = tag + '/Eval/{}'
        self._summary_writer = None
        self._checkpoint_path = Path(checkpoint_dir) / checkpoint_name
        self._saver = None

        self._timer = SecondOrStepTimer(every_steps=evaluator.eval_steps)
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        self._steps_pre_run = steps_pre_run
        self._need_save = False
        self._better_result = None

        if self._get_best_result_dump_file().exists():
            self._better_result = yaml_tools.read(self._get_best_result_dump_file())
            tf.logging.info("load completed best result records")

    def begin(self):
        self._summary_writer = SummaryWriterCache.get(self.checkpoint_dir)

    def before_run(self, run_context):
        return SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(stale_global_step + self._steps_pre_run):
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                if self._evaluate(run_context.session, global_step):
                    run_context.request_stop()

    def end(self, session):
        last_step = session.run(self._global_step_tensor)
        if last_step != self._timer.last_triggered_step():
            self._evaluate(session, last_step)

    def _get_best_result_dump_file(self, name="best_result"):  # todo test 之后再改?
        return Path(self._checkpoint_path).parent / name

    def _get_result(self):
        result = dict()
        for key, value in self._better_result.items():
            if isinstance(value, np.int64):
                value = int(value)
            else:
                value = float(value)
            result[key] = value

        return result

    def _evaluate(self, session, step):
        results = self.evaluator.evaluate_with_session(session)
        if not self._better_result or self.evaluator.compare(results, self._better_result):
            self._better_result = results
            self._need_save = True

        self._summary(step, results)
        return self._save(session, step)

    def _summary(self, step, result=None):
        if result is None:
            result = self._better_result

        tags, values = [], []
        for key, value in result.items():
            if key == CustomKeys.GLOBAL_STEP:
                continue
            tags.append(self._summary_tag.format(key))
            values.append(value)
        summary_tools.scalar(self._summary_writer, step, tags, values)

    def _save(self, session, step):
        """Saves the better checkpoint, returns should_stop."""
        if not self._need_save:
            return False
        self._need_save = False
        self._get_saver().save(session, self._checkpoint_path, global_step=step,
                               latest_filename="checkpoint_best")
        yaml_tools.write(self._get_best_result_dump_file(), self._get_result())
        should_stop = False
        tf.logging.info("saved best checkpoints for {} into {}".format(step, str(self._checkpoint_path)))
        return should_stop

    def _get_saver(self):
        if self._saver is not None:
            return self._saver
        # Get saver from the SAVERS collection if present.
        savers = ops.get_collection(CustomKeys.SAVERS)
        self._saver = saver_lib.Saver(saver_def=savers[0].as_saver_def())
        return self._saver
