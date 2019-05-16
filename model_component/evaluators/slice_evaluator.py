from model_component.evaluators.base import *


class SliceEvaluator(BaseEvaluate):
    name = 'slice'

    def set_init_config(self, params):
        self.eval_steps = params['eval_steps']
        self.metric_list = params['metric_list']
        self.save_image = params['save_image']
        self.show_each_evaluate = params['show_each_evaluate']

    def set_config(self, **params):
        tf.logging.info('..... setting {} evaluate config'.format(self.name))
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
            predictions.update(value.get_predictions())
        predict_generation = self._predict_with_session(session, predictions, None)
        self.clear_metrics()
        for prediction in predict_generation:
            prediction = flat_dict_convert_solid_dict(prediction)
            for class_key, class_value in prediction[CustomKeys.CLASSES].items():
                for metric_key, metric_value in class_value[CustomKeys.METRICS].items():
                    self.append_metrics({'{}/{}'.format(class_key, metric_key): metric_value})
        results = {key: np.mean(values) for key, values in self._metric_values.items()}
        for key, value in results.items():
            tf.logging.info('{}----->{:.3f}\t'.format(key, value))
        return results

    def set_eval_config(self, params):
        from model_component.estimator import MyEstimator
        self.estimator = MyEstimator()
        params['evaluator'] = self
        self.estimator.set_eval_config(params)
        self.dataset = self.estimator.dataset

    def evaluate(self):
        tf.logging.info('begin {} evaluating......................................................'.format(self.name))
        predict_generation = self.estimator.get_evaluate_predictions()
        self.clear_metrics()
        for prediction in predict_generation:
            self.append_metrics(prediction)
        results = {key: np.mean(values) for key, values in self._metric_values.items()}
        for key, value in results.items():
            tf.logging.info('{}----->{:.3f}\t'.format(key, value))
