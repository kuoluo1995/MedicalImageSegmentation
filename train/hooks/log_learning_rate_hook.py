from tensorflow.contrib.learn.python.learn.summary_writer_cache import SummaryWriterCache
from tensorflow.python.framework import ops
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunHook, SessionRunArgs
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from utils import summary_tools
from train.config import CustomKeys


class LogLearningRateHook(SessionRunHook):

    def __init__(self, every_steps, steps_per_run, output_dir, tag):
        self._timer = SecondOrStepTimer(every_steps=every_steps)
        self.steps_per_run = steps_per_run
        self._summary_writer = SummaryWriterCache.get(output_dir)
        self.summary_tag = '{}/learning rate'.format(tag)
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        self._learning_rate = None

    def after_create_session(self, session, coord):
        self._learning_rate = ops.get_collection(CustomKeys.LEARNING_RATE)[0]

    def before_run(self, run_context):
        return SessionRunArgs([self._learning_rate, self._global_step_tensor])

    def after_run(self, run_context, run_values):
        learning_rate, global_step = run_values.results
        if self._timer.should_trigger_for_step(global_step + self.steps_per_run):
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                self._log_and_record(learning_rate, global_step)

    def _log_and_record(self, learning_rate, step):
        if self._summary_writer is not None:
            summary_tools.scalar(self._summary_writer, step, [self.summary_tag], [learning_rate])
