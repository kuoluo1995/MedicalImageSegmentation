from train.evaluators.slice_evaluator import SliceEvaluator


def create_evaluator(evaluator_name):
    class_instance = eval(evaluator_name)()
    return class_instance
