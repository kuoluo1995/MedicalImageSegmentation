from train.mode.eval_mode import EvalMode
from train.mode.train_mode import TrainMode
from train.mode.eval_mode_3d import EvalMode3D


def create_mode(mode_name):
    class_instance = eval(mode_name)()
    return class_instance
