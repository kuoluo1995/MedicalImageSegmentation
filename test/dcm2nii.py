import matplotlib.pyplot as plt
import nibabel
import numpy as np
import SimpleITK as sitk
import os
from pathlib import Path

output_path = Path('E:/Corpus_Callosum/')
output_path.mkdir(parents=True, exist_ok=True)
reader = sitk.ImageSeriesReader()
remove_files = list()
remove_folds = list()


def deal_source_image():
    for fold in Path('E:/CorpusCallosum/AD').iterdir():
        output_path = Path('E:/Corpus_Callosum/' + fold.stem.replace(' ', '_'))
        if fold.is_dir():
            label = str(fold).replace('AD', 'ADCCROI') + '_Merge.nii'
            label = nibabel.load(str(label))

            dicoms = reader.GetGDCMSeriesFileNames(str(fold))
            reader.SetFileNames(dicoms)
            image = reader.Execute()
            print(fold.stem + ' image:' + str(image.GetSize()) + ' label:' + str(label.shape))

            if image.GetSize() != label.shape or label.shape[2] != 256:
                output_path = Path('E:/Corpus_Callosum/check/' + fold.stem.replace(' ', '_'))
            output_path.mkdir(parents=True, exist_ok=True)
            nibabel.save(label, str(output_path) + '/label.nii')
            sitk.WriteImage(image, str(output_path) + '/image.nii')
        else:
            label = str(fold).replace('AD', 'ADCCROI').replace('.nii', '_Merge.nii')
            label = nibabel.load(str(label))

            image = nibabel.load(str(fold))
            print(fold.stem + ' image:' + str(image.shape) + ' label:' + str(label.shape))

            if image.shape != label.shape or label.shape[2] != 256:
                output_path = Path('E:/Corpus_Callosum/check/' + fold.stem.replace(' ', '_'))
            output_path.mkdir(parents=True, exist_ok=True)
            nibabel.save(label, str(output_path) + '/label.nii')
            nibabel.save(image, str(output_path) + '/image.nii')


def same_shape_z192():
    # 处理 255，255，192的图片 shepe相同
    for fold in Path('E:/Corpus_Callosum/check').iterdir():
        label = nibabel.load(str(fold) + '/label.nii')
        label_array = label.get_fdata()
        image = nibabel.load(str(fold) + '/image.nii')
        image_array = image.get_fdata()
        if image.shape == label.shape:
            image_array = np.transpose(image_array, (2, 1, 0))
            image_header = image.header
            image_affine = image.affine

            new_image = nibabel.Nifti1Image(image_array, affine=image_affine, header=image_header)
            plt.imshow(image_array[102])
            plt.pause(0.001)

            label_array = np.transpose(label_array, (2, 1, 0))
            label_header = label.header
            label_affine = label.affine

            new_label = nibabel.Nifti1Image(label_array, affine=label_affine, header=label_header)
            plt.imshow(label_array[102])
            plt.pause(0.001)

            # print('image:' + str(image_array.shape) + ' label:' + str(label_array.shape))
            output_path = Path('E:/Corpus_Callosum/checked/' + fold.stem.replace(' ', '_'))
            output_path.mkdir(parents=True, exist_ok=True)
            nibabel.save(new_label, str(output_path) + '/label.nii')
            nibabel.save(new_image, str(output_path) + '/image.nii')

            remove_files.append(str(fold) + '/label.nii')
            remove_files.append(str(fold) + '/image.nii')
            remove_folds.append(str(fold))

    for file in remove_files:
        os.remove(file)
    for fold in remove_folds:
        os.removedirs(fold)


def same_shape_x192():
    # 处理192，255，255的图片 shape相同
    for fold in Path('E:/Corpus_Callosum').iterdir():
        if fold.stem == 'check' or fold.stem == 'checked':
            continue
        label = nibabel.load(str(fold) + '/label.nii')
        label_array = label.get_fdata()
        image = nibabel.load(str(fold) + '/image.nii')
        image_array = image.get_fdata()
        if image.shape == label.shape:
            image_array = np.transpose(image_array, (0, 2, 1))
            image_header = image.header
            image_affine = image.affine
            for i in range(image_array.shape[0]):
                image_array[i] = np.rot90(image_array[i], 2)
            new_image = nibabel.Nifti1Image(image_array, affine=image_affine, header=image_header)
            plt.imshow(image_array[102])
            plt.pause(0.001)

            label_array = np.transpose(label_array, (0, 2, 1))
            label_header = label.header
            label_affine = label.affine
            for i in range(label_array.shape[0]):
                label_array[i] = np.rot90(label_array[i], 2)
            new_label = nibabel.Nifti1Image(label_array, affine=label_affine, header=label_header)
            plt.imshow(label_array[102])
            plt.pause(0.001)

            output_path = Path('E:/Corpus_Callosum/checked/' + fold.stem.replace(' ', '_'))
            output_path.mkdir(parents=True, exist_ok=True)
            nibabel.save(new_label, str(output_path) + '/label.nii')
            nibabel.save(new_image, str(output_path) + '/image.nii')

            remove_files.append(str(fold) + '/label.nii')
            remove_files.append(str(fold) + '/image.nii')
            remove_folds.append(str(fold))

    for file in remove_files:
        os.remove(file)
    for fold in remove_folds:
        os.removedirs(fold)


def different_shape():
    # shape不同的
    for fold in Path('E:/Corpus_Callosum/check').iterdir():
        label = nibabel.load(str(fold) + '/label.nii')
        label_array = label.get_fdata()
        image = nibabel.load(str(fold) + '/image.nii')
        image_array = image.get_fdata()
        if image.shape != label.shape:

            image_array = np.transpose(image_array, (2, 1, 0))
            image_header = image.header
            image_affine = image.affine
            new_image = nibabel.Nifti1Image(image_array, affine=image_affine, header=image_header)
            plt.imshow(image_array[102])
            plt.pause(0.001)

            label_header = label.header
            label_affine = label.affine
            for i in range(label_array.shape[0]):
                label_array[i] = np.fliplr(np.rot90(label_array[i], 1))
            new_label = nibabel.Nifti1Image(label_array, affine=label_affine, header=label_header)
            plt.imshow(label_array[102])
            plt.pause(0.001)

            output_path = Path('E:/Corpus_Callosum/checked/' + fold.stem.replace(' ', '_'))
            output_path.mkdir(parents=True, exist_ok=True)
            nibabel.save(new_label, str(output_path) + '/label.nii')
            nibabel.save(new_image, str(output_path) + '/image.nii')

            remove_files.append(str(fold) + '/label.nii')
            remove_files.append(str(fold) + '/image.nii')
            remove_folds.append(str(fold))

    for file in remove_files:
        os.remove(file)
    for fold in remove_folds:
        os.removedirs(fold)


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
