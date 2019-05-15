from build_dataset.base import *


class CarcassDataset(BaseDataset):
    name = 'Carcass'

    # *************************************************创建时用的函数************************************************* #
    def set_build_config(self, params):
        dataset_config = params['dataset']
        self._source_data_path = Path(dataset_config['source_data']['path'])
        self._output_data_path = Path(__file__).parent.parent / 'dataset' / dataset_config['name']
        self._output_data_path.mkdir(parents=True, exist_ok=True)
        self.train_scale = dataset_config['output_data']['train_scale']
        self.seed = params['random_seed']
        self.k = dataset_config['output_data']['k']
        self._k_folds_record = self._output_data_path / '_k_folds_record.yaml'  # 固定值，内部文件
        self._examples = dataset_config['examples']['value']
        tf.logging.info('set completed {} dataset config'.format(self.name))

    def _get_source_data(self):
        source_data = list()
        for fold in self._source_data_path.iterdir():
            source_data.append({'image': str(fold) + '/image.nii', 'label': str(fold) + '/label.nii'})
        return source_data

    @staticmethod
    def deal_image(image_reader):
        return image_reader

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
