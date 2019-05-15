from evaluators.slice_evaluator import SliceEvaluator
from evaluators.volume_evaluator import VolumeEvaluator


def create_evaluator(evaluator_name):
    class_instance = eval(evaluator_name)()
    return class_instance
