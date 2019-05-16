from model_component.mode.eval_mode import EvalMode
from model_component.mode.train_mode import TrainMode
from model_component.mode.eval_mode_3d import EvalMode3D
from model_component.mode.train_organ_mode import TrainOrganMode


def create_mode(mode_name):
    class_instance = eval(mode_name)()
    return class_instance
