from train.mode.eval_mode import EvalMode
from train.mode.train_mode import TrainMode
from train.mode.train_organ_mode import TrainOrganMode


def create_mode(mode_name):
    class_instance = eval(mode_name)()
    return class_instance
