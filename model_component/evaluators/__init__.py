from model_component.evaluators.slice_evaluator import SliceEvaluator
from model_component.evaluators.volume_evaluator import VolumeEvaluator


def create_evaluator(evaluator_name, params):
    class_instance = eval(evaluator_name)()
    class_instance.set_init_config(params)
    return class_instance
