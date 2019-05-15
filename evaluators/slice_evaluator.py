from evaluators.base import *


class SliceEvaluator(BaseEvaluate):
    name = 'slice'

    def set_config(self, **params):
        tf.logging.info('..... setting {} evaluate config'.format(self.name))
        self.eval_steps = params['eval_steps']
        self.metric_list = params['metric_list']
        self.model_dict = params['model_dict']
        self.estimator = params['estimator']
        self.save_image = params['save_image']
        self.show_each_evaluate = params['show_each_evaluate']

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
            self.append_metrics(prediction)
        results = {key: np.mean(values) for key, values in self._metric_values.items()}
        for key, value in results.items():
            tf.logging.info('{}----->{:.3f}\t'.format(key, value))
        return results

    def set_eval_config(self, params):
        self.eval_steps = params['evaluator']['eval_steps']
        self.metric_list = params['evaluator']['metric_list']
        self.save_image = params['evaluator']['save_image']
        self.show_each_evaluate = params['evaluator']['show_each_evaluate']
        self.estimator = train.estimator.MyEstimator()
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
