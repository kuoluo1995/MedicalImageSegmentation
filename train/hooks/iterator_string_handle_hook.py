import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook


class IteratorStringHandleHook(SessionRunHook):
    def __init__(self, mode_dict):
        self.mode_dict = mode_dict

    def get_handle(self, key):
        return self.mode_dict[key]

    def begin(self):
        for key in self.mode_dict:
            self.mode_dict[key].string_handle = self.mode_dict[key].iterator.string_handle()

    def after_create_session(self, session, coord):
        del coord
        for key in self.mode_dict:
            self.mode_dict[key].handler = session.run(self.mode_dict[key].string_handle)
        tf.logging.info("build completed iterator handler")
