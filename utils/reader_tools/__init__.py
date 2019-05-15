import nibabel
import SimpleITK
from utils.reader_tools.image_reader import ImageReader
from utils.reader_tools.volume_reader import VolumeReader


def create_reader(reader_name, **params):
    class_instance = eval(reader_name)()
    class_instance.set_config(params)
    return class_instance


def mhd_reader(path):
    itkimage = SimpleITK.ReadImage(str(path))
    # z, y, x
    mr_scan = SimpleITK.GetArrayFromImage(itkimage)
    return mr_scan


def mhd_header_reader(path):
    itkimage = SimpleITK.ReadImage(str(path))
    # x, y, z
    origin = itkimage.GetOrigin()
    # x, y, z
    spacing = itkimage.GetSpacing()
    return {'origin': origin, 'spacing': spacing}


def mhd_writer(path, header, image_array):
    itkimage = SimpleITK.GetImageFromArray(image_array, isVector=False)
    itkimage.SetSpacing(header['spacing'])
    itkimage.SetOrigin(header['origin'])
    SimpleITK.WriteImage(itkimage, path, False)


def nii_reader(path):
    image = nibabel.load(str(path))
    image_array = image.get_fdata()
    return image_array


def nii_header_reader(path):
    image = nibabel.load(str(path))
    image_header = image.header
    image_affine = image.affine
    return {'header': image_header, 'affine': image_affine}


def nii_writer(path, header, image_array):
    image = nibabel.Nifti1Image(image_array, affine=header['affine'], header=header['header'])
    nibabel.save(image, str(path))
