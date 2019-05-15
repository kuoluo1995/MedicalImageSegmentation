from pathlib import Path

from evaluators.base import *


class VolumeEvaluator(BaseEvaluate):
    name = 'volume'
    image_reader = VolumeReader()

    def set_config(self, **params):
        tf.logging.info('..... setting {} evaluate config'.format(self.name))
        self.eval_steps = params['eval_steps']
        self.metric_list = params['metric_list']
        self.model_dict = params['model_dict']
        self.estimator = params['estimator']
        self.dataset = params['dataset']
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

    def _evaluate(self, predict_generation):
        def evaluate_case():
            logits3d = {class_key: np.concatenate(values) for class_key, values in logits.items()}
            labels3d = {class_key: np.concatenate(values) for class_key, values in labels.items()}
            image_path = Path(current_path)
            self.image_reader.read_header_dict(image_path)
            metrics_dict = dict()
            for class_key, image3d in logits3d.items():
                metrics_3d = metrics_function.metric_3d(logits3d[class_key], labels3d[class_key],
                                                        required=self.metric_list,
                                                        sampling=self.image_reader.header['spacing'])
                for metric_key, metric_value in metrics_3d.items():
                    metrics_dict['{}/{}'.format(class_key, metric_key)] = metric_value
            self.append_metrics(metrics_dict)

            if self.show_each_evaluate:
                for key_, value_ in metrics_dict.items():
                    tf.logging.info('<- {} ->  {} = {:.3f}'.format(current_path, key_, value_))

            if self.save_image:
                for class_key in logits3d:
                    save_path = image_path.parent / (class_key + '_' + CustomKeys.PREDICTION + image_path.suffix)
                    # todo improve 图像融合
                    prediction = self.dataset.restore_image(np.squeeze(logits3d[class_key]))
                    self.image_reader.save(prediction, save_path)
                    tf.logging.info('================> saved predictions at {}'.format(str(save_path)))
            return metrics_dict

        logits = defaultdict(list)
        labels = defaultdict(list)
        current_path = None
        current_step = 0
        for predict in predict_generation:
            predict = flat_dict_convert_solid_dict(predict)
            new_path = predict['image_path'][0].decode('utf-8')  # batch_size个name
            if current_path is None:
                current_path = new_path
            if current_path == new_path:
                for key, value in predict[CustomKeys.CLASSES].items():
                    logits[key].append(value[CustomKeys.PREDICTION])
                    labels[key].append(value[CustomKeys.LABEL])
            else:
                evaluate_case()
                for key in predict[CustomKeys.CLASSES]:
                    logits[key].clear()
                    labels[key].clear()
                current_path = new_path
                if current_step >= self.eval_steps:
                    break
        logits.clear()
        labels.clear()

    def evaluate_with_session(self, session, cases=None):
        tf.logging.info('begin {} evaluating......................................................'.format(self.name))
        predictions = dict()
        for key, value in self.model_dict.items():
            predictions.update(value.get_predictions())
        predict_generation = self._predict_with_session(session, predictions, None)
        self.clear_metrics()
        self._evaluate(predict_generation)
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
        self._evaluate(predict_generation)
        results = {key: np.mean(values) for key, values in self._metric_values.items()}
        for key, value in results.items():
            tf.logging.info('{}----->{:.3f}\t'.format(key, value))
