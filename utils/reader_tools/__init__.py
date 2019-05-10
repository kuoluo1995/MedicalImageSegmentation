import nibabel
import SimpleITK
from utils.reader_tools.section_reader import ImageReader


def create_reader(reader_name, image_type, is_label):
    class_instance = eval(reader_name)(image_type, is_label)
    return class_instance


def mhd_reader(path):
    itkimage = SimpleITK.ReadImage(str(path))
    # z, y, x
    mr_scan = SimpleITK.GetArrayFromImage(itkimage)
    # # x, y, z
    # origin = itkimage.GetOrigin()
    # # x, y, z
    # spacing = itkimage.GetSpacing()
    return mr_scan


def nii_reader(path):
    image = nibabel.load(str(path))
    image_array = image.get_fdata()
    image_header = image.header
    image_affine = image.affine
    return image_array


def mhd_writer(path, image, **kwargs):
    SimpleITK.WriteImage(image, path, False)


def nii_writer(path, image_array, header, affine, **kwargs):
    image = nibabel.Nifti1Image(image_array, affine=affine, header=header)
    nibabel.save(image, str(path))
