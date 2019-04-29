import SimpleITK


def mhd_reader(path):
    itkimage = SimpleITK.ReadImage(str(path))
    # z, y, x
    mr_scan = SimpleITK.GetArrayFromImage(itkimage)
    # # x, y, z
    # origin = itkimage.GetOrigin()
    # # x, y, z
    # spacing = itkimage.GetSpacing()
    return mr_scan


def mhd_writer(path, image):
    itkimage = SimpleITK.GetArrayFromImage(image)
    # itkimage.SetSapcing(spacing)
    # itkimage.SetOrigin(origin)
    SimpleITK.WriteImage(image, path, False)
