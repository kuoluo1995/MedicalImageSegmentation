from train.evaluators.base import *


class SliceEvaluator(BaseEvaluate):
    name = 'slice'

    def set_config(self, **params):
        tf.logging.info('..... setting {} evaluate config'.format(self.name))
        self.eval_steps = params['eval_steps']
        self.metric_list = params['metric_list']
        self.model_dict = params['model_dict']
        self.estimator = params['estimator']

    def compare(self, *args, **kwargs):
        def _compare(current_result, origin_result, **kwargs):
            for name in self.metric_list:
                for key in current_result:
                    if name in key:
                        if current_result[key] == origin_result[key]:
                            continue
                        return True if current_result[key] > origin_result[key] else False
            return False

        return _compare(*args, **kwargs)

    def evaluate_with_session(self, session):
        tf.logging.info('begin {} evaluating......................................................'.format(self.name))
        predictions = dict()
        for key, value in self.model_dict.items():
            predictions.update(value.get_metrics_value())
        self.clear_metrics()
        predict_generation = self._predict_with_session(session, predictions, None)
        for prediction in predict_generation:
            self.append_metrics(prediction)
        results = {key: np.mean(values) for key, values in self._metric_values.items()}
        display = ''
        for key, value in results.items():
            display += '{}----->{:.3f}\t'.format(key, value)
        tf.logging.info(display)
        return results
