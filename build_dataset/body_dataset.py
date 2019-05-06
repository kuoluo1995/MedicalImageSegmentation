from build_dataset.base import *


class BodyDataset(BaseDataset):
    name = 'Body'
    _image_pattern = None

    # *************************************************创建时用的函数************************************************* #
    def set_build_config(self, params):
        dataset_config = params['dataset']
        self._source_data_path = Path(dataset_config['source_data']['path'])
        self._output_data_path = Path(__file__).parent.parent / 'dataset' / dataset_config['name']
        if not self._output_data_path.exists():
            self._output_data_path.mkdir(parents=True, exist_ok=True)
        self.train_scale = dataset_config['output_data']['train_scale']
        self._image_pattern = dataset_config['source_data']['extra']['image_pattern']
        self.seed = params['random_seed']
        self.k = dataset_config['output_data']['k']
        self._k_folds_record = self._output_data_path / '_k_folds_record.yaml'  # 固定值，内部文件
        self._examples = dataset_config['examples']['value']
        tf.logging.info('set completed {} dataset config'.format(self.name))

    def _get_source_data(self):
        image_data = list(self._source_data_path.rglob(self._image_pattern))
        source_data = [
            {'image': str(source).replace('stir', 'STIR'), 'label': str(source).replace('stir.mhd', 'STIR-label.mhd')}
            for source in image_data]
        return source_data

    # *************************************************训练时用的函数************************************************* #
    def set_train_config(self, **param):
        tf.logging.info('..... setting {} train config'.format(self.name))
        self.batch_size = param['batch_size']
        self._output_data_path = Path(__file__).parent.parent / 'dataset' / param['name']
        self.seed = param['random_seed']
        self.min_window_level = param['min_window_level']
        self.max_window_level = param['max_window_level']
        self.image_width = param['image_width']
        self.image_height = param['image_height']
        self.num_parallel_batches = param['num_parallel_batches']
        self.image_augmentation_dict = param['image_augmentation']
