import numpy as np
import matplotlib.pyplot as plt
import nibabel
import SimpleITK as sitk
from pathlib import Path

output_path = Path('E:/Corpus_Callosum/')
output_path.mkdir(parents=True, exist_ok=True)
reader = sitk.ImageSeriesReader()
remove_files = list()
remove_folds = list()
image_max = 5000


def deal_source_image():
    for fold in Path('E:/source_data').iterdir():
        max_window_level = int(fold.name)
        for people_fold in fold.iterdir():
            label = nibabel.load(str(people_fold) + '/label.nii')
            image = nibabel.load(str(people_fold) + '/image.nii')
            image_header = image.header
            image_affine = image.affine
            array = image.get_fdata()
            array /= max_window_level
            array = np.clip(array, 0, 1) * image_max
            image = nibabel.Nifti1Image(array, affine=image_affine, header=image_header)
            path = output_path / people_fold.name
            path.mkdir(parents=True, exist_ok=True)
            nibabel.save(label, str(path) + '/label.nii')
            nibabel.save(image, str(path) + '/image.nii')


def show_nii():
    for fold in Path('E:/Corpus_Callosum/checked').iterdir():
        label = nibabel.load(str(fold) + '/label.nii')
        label_array = label.get_fdata()
        image = nibabel.load(str(fold) + '/image.nii')
        image_array = image.get_fdata()

        plt.imshow(image_array[102])
        plt.pause(0.001)

        plt.imshow(label_array[102])
        plt.pause(0.001)


deal_source_image()
