import tensorflow as tf
from pathlib import Path
from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import ops, random_seed
from tensorflow.python.training import training, warm_starting_util

import build_dataset
from train import config, evaluators, hooks, mode, networks
from train.config import CustomKeys
from train.solver.solver import Solver


class MyEstimator(object):
    tag = None
    handler = None
    seed = None
    batch_size = None
    num_steps = None
    _model_dir = None
    _config = None

    _model_dict = dict()
    # checkpoint = None

    mode_dict = dict()
    _dataset = None
    _solver = None

    _feed_dict = dict()
    _init_fn = None
    _estimator_spec = None

    def _get_input_pipeline(self):
        with tf.variable_scope('InputPipeline'):
            (feature, label), input_hooks, handler = self._dataset.get_data_iterator(self.mode_dict)
            return feature, label, input_hooks, handler

    def _get_model(self, feature, label):
        with tf.variable_scope('Model'):
            loss = None  # todo improve 这里未来可以考虑多个模型
            predictions = None
            for _, model in self._model_dict.items():
                loss = model.build_model(feature, label)
                predictions = model.get_predictions()
            train_optimizer = self._solver.get_train_optimizer(loss)
            init_fn = self._get_scaffold_init()
            scaffold = tf.train.Scaffold(init_fn=init_fn)
            kwargs = {'loss': loss, 'train_op': train_optimizer, 'predictions': predictions, 'scaffold': scaffold}
            return tf.estimator.EstimatorSpec(mode=model_fn_lib.ModeKeys.TRAIN, **kwargs)
            # todo read 这个函数预测时可能要再研究下

    def _get_hooks_args(self, model):
        tensors = {'loss': model.loss, 'step': tf.train.get_or_create_global_step()}
        tensors.update({key: val for key, val in model.predictions.items() if '_metric' in key})
        # todo improve 待优化hooks这种结构
        hooks_args = {'model': model, 'num_steps': self.num_steps, 'tensors': tensors,
                      'config': self._config, 'mode_dict': self.mode_dict, 'tag': self.tag,
                      'evaluator': self._evaluator, 'model_dir': str(self._model_dir),
                      'every_steps': self._evaluator.eval_steps, 'steps_pre_run': 1}
        return hooks_args

    # *************************************************训练时用的函数************************************************* #
    def set_train_config(self, params):
        random_seed.set_random_seed(params['random_seed'])
        self.seed = params['random_seed']
        self.tag = params['tag']
        self.batch_size = params['batch_size']
        self.num_steps = params['estimator']['num_steps']
        self._model_dir = Path(__file__).parent.parent / params['model']['model_dir'] / self.tag
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._warm_start_settings = estimator_lib._get_default_warm_start_settings(None)
        self._config = estimator_lib.maybe_overwrite_model_dir_and_session_config(
            tf.estimator.RunConfig(
                tf_random_seed=params['random_seed'],
                session_config=config.get_session_config(params['estimator']['allow_soft_placement'],
                                                         params['estimator']['gpu_options_allow_growth']),
                keep_checkpoint_max=params['estimator']['keep_checkpoint_max'],
                log_step_count_steps=params['estimator']['log_step_count_steps']
            ), str(self._model_dir))
        self._set_modes(params)
        self._set_dataset(params)
        self._set_models(params)
        self._set_solver(params)
        self._set_evaluator(params)

    def train(self):
        tf.logging.info('>>>>>>>>>>>>>>>>>>>> building input pipeline')
        feature, label, input_hooks, self.handler = self._get_input_pipeline()

        tf.logging.info('>>>>>>>>>>>>>>>>>>>> building model')
        model = self._get_model(feature, label)

        if self._warm_start_settings:
            tf.logging.info('>>>>>>>>>>>>>>>>>>>> building warm start')
            warm_starting_util.warm_start(*self._warm_start_settings)

        # Create Saver object
        tf.logging.info('>>>>>>>>>>>>>>>>>>>> building saver')
        self._create_saver(model)

        with tf.variable_scope('AddHooks'):
            tf.logging.info('>>>>>>>>>>>>>>>>>>>> building hooks')
            hooks_args = self._get_hooks_args(model)
            work_hooks = hooks.get_work_hooks(hooks_args, input_hooks, model.training_chief_hooks)

        with tf.variable_scope('Training'):
            tf.logging.info('>>>>>>>>>>>>>>>>>>>> building training')
            with training.MonitoredTrainingSession(
                    master=self._config.master,
                    is_chief=self._config.is_chief,
                    checkpoint_dir=str(self._model_dir),
                    scaffold=model.scaffold,
                    hooks=work_hooks,
                    chief_only_hooks=hooks.get_chief_hooks_tuple(hooks_args, model.training_chief_hooks),
                    save_summaries_steps=self._config.save_summary_steps,
                    config=self._config.session_config,
                    log_step_count_steps=self._config.log_step_count_steps
            ) as session:
                self._feed_dict[self.handler] = self.mode_dict['TrainMode'].handler
                while not session.should_stop():
                    _, loss = session.run([model.train_op, model.loss], self._feed_dict)

    def _set_modes(self, params):
        tf.logging.info('>>>>>>>>>>>>>>>>>>>> setting modes')
        for key, value in params['dataset']['modes'].items():
            self.mode_dict[key] = mode.create_mode(key)
            self.mode_dict[key].set_config(**value)

    def _set_dataset(self, params):
        tf.logging.info('>>>>>>>>>>>>>>>>>>>> setting dataset')
        self._dataset = build_dataset.create_dataset(params['dataset']['name'])
        self._dataset.set_train_config(**params['dataset'], batch_size=self.batch_size, random_seed=self.seed)

    def _set_models(self, params):
        tf.logging.info('>>>>>>>>>>>>>>>>>>>> setting models')
        for key, value in params['model']['networks'].items():
            self._model_dict[key] = networks.create_network(key)
            self._model_dict[key].set_config(**value, **params['model'], tag=self.tag, batch_size=self.batch_size,
                                             image_channel=self._dataset.image_channel,
                                             image_height=params['dataset']['image_height'],
                                             image_width=params['dataset']['image_width']
                                             )

    def _set_solver(self, params):
        tf.logging.info('>>>>>>>>>>>>>>>>>>>> setting solver')
        self._solver = Solver()
        self._solver.set_config(params['model']['solver'])

    def _set_evaluator(self, params):
        tf.logging.info('>>>>>>>>>>>>>>>>>>>> setting evaluator')
        self._evaluator = evaluators.create_evaluator(params['model']['evaluator']['name'])
        self._evaluator.set_config(model_dict=self._model_dict, estimator=self, **params['model']['evaluator'])

    def _get_scaffold_init(self):
        # todo improve 改善
        return None

    def _create_saver(self, model):
        if not (model.scaffold.saver or ops.get_collection(CustomKeys.SAVERS)):
            ops.add_to_collection(CustomKeys.SAVERS,
                                  training.Saver(sharded=True, max_to_keep=self._config.keep_checkpoint_max,
                                                 keep_checkpoint_every_n_hours=(
                                                     self._config.keep_checkpoint_every_n_hours), defer_build=True,
                                                 save_relative_paths=True))

    # *************************************************评估时用的函数************************************************* #
    def set_eval_config(self, params):
        random_seed.set_random_seed(params['random_seed'])
        self.seed = params['random_seed']
        self.tag = params['tag']
        self.batch_size = params['batch_size']
        self._model_dir = Path(__file__).parent.parent / params['model']['model_dir'] / self.tag
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._warm_start_settings = estimator_lib._get_default_warm_start_settings(None)
        self._config = estimator_lib.maybe_overwrite_model_dir_and_session_config(
            tf.estimator.RunConfig(tf_random_seed=params['random_seed'],
                                   session_config=config.get_session_config(params['estimator']['allow_soft_placement'],
                                                                            params['estimator'][
                                                                                'gpu_options_allow_growth'])),
            str(self._model_dir))
        self._set_modes(params)
        self._set_dataset(params)
        self._set_models(params)
        self._set_evaluator(params)

    def evaluate(self):
        tf.logging.info('>>>>>>>>>>>>>>>>>>>> building input pipeline')
        feature, label, input_hooks, self.handler = self._get_input_pipeline()
        tf.logging.info('>>>>>>>>>>>>>>>>>>>> building model')
        model = self._get_model(feature, label)
        with tf.variable_scope('AddHooks'):
            tf.logging.info('>>>>>>>>>>>>>>>>>>>> building hooks')
            work_hooks = hooks.get_work_hooks(None, input_hooks, model.training_chief_hooks)

        with tf.variable_scope('Training'):
            tf.logging.info('>>>>>>>>>>>>>>>>>>>> building training')
            with training.MonitoredSession(
                    session_creator=training.ChiefSessionCreator(
                        master=self._config.master,
                        scaffold=model.scaffold,
                        config=self._config.session_config),
                    hooks=work_hooks) as session:
                while not session.should_stop():
                    _, loss = session.run([model.train_op, model.loss], self._feed_dict)
