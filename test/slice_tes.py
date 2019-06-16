import vtk
from vtkmodules.vtkCommonColorPython import vtkNamedColors
from vtkmodules.vtkCommonCorePython import vtkLookupTable
from vtkmodules.vtkFiltersCorePython import vtkMarchingCubes, vtkStripper
from vtkmodules.vtkFiltersModelingPython import vtkOutlineFilter
from vtkmodules.vtkIOImagePython import vtkNIFTIImageReader
from vtkmodules.vtkImagingCorePython import vtkImageMapToColors
from vtkmodules.vtkRenderingCorePython import vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor, vtkCamera, \
    vtkImageActor, vtkPolyDataMapper, vtkActor


def main():
    colors = vtkNamedColors()
    fileName = 'E:/Corpus_Callosum/Achen_caixing/image.nii'
    colors.SetColor("SkinColor", 255, 125, 64, 255)
    colors.SetColor("BkgColor", 255, 255, 255, 255)

    renderer = vtkRenderer()
    renWin = vtkRenderWindow()
    renWin.AddRenderer(renderer)
    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renWin)

    # Set a background color for the renderer and set the size of the
    # render window (expressed in pixels).
    renderer.SetBackground(colors.GetColor3d("BkgColor"))
    renWin.SetSize(640, 480)

    # The following reader is used to read a series of 2D slices (images)
    # that compose the volume. The slice dimensions are set, and the
    # pixel spacing. The data Endianness must also be specified. The
    # reader uses the FilePrefix in combination with the slice number to
    # construct filenames using the format FilePrefix.%d. (In this case
    # the FilePrefix is the root name of the file: quarter.)
    reader = vtkNIFTIImageReader()
    reader.SetFileName(fileName)
    reader.Update()

    # An isosurface, or contour value of 500 is known to correspond to
    # the skin of the patient.
    # The triangle stripper is used to create triangle
    # strips from the isosurface these render much faster on may
    # systems.
    skinExtractor = vtkMarchingCubes()
    skinExtractor.SetInputConnection(reader.GetOutputPort())
    skinExtractor.SetValue(0, 1000)
    skinExtractor.Update()

    skinStripper = vtkStripper()
    skinStripper.SetInputConnection(skinExtractor.GetOutputPort())
    skinStripper.Update()

    skinMapper = vtkPolyDataMapper()
    skinMapper.SetInputConnection(skinStripper.GetOutputPort())
    skinMapper.ScalarVisibilityOff()

    skin = vtkActor()
    skin.SetMapper(skinMapper)
    skin.GetProperty().SetDiffuseColor(colors.GetColor3d("SkinColor"))
    skin.GetProperty().SetSpecular(.3)
    skin.GetProperty().SetSpecularPower(20)

    # An isosurface, or contour value of 1150 is known to correspond to
    # the bone of the patient.
    # The triangle stripper is used to create triangle
    # strips from the isosurface these render much faster on may
    # systems.
    boneExtractor = vtkMarchingCubes()
    boneExtractor.SetInputConnection(reader.GetOutputPort())
    boneExtractor.SetValue(0, 5000)

    boneStripper = vtkStripper()
    boneStripper.SetInputConnection(boneExtractor.GetOutputPort())

    boneMapper = vtkPolyDataMapper()
    boneMapper.SetInputConnection(boneStripper.GetOutputPort())
    boneMapper.ScalarVisibilityOff()

    bone = vtk.vtkActor()
    bone.SetMapper(boneMapper)
    bone.GetProperty().SetDiffuseColor(colors.GetColor3d("Ivory"))

    # Now we are creating three orthogonal planes passing through the
    # volume. Each plane uses a different texture map and therefore has
    # different coloration.

    # Start by creating a black/white lookup table.
    bwLut = vtkLookupTable()
    bwLut.SetTableRange(0, 2000)
    bwLut.SetSaturationRange(0, 0)
    bwLut.SetHueRange(0, 0)
    bwLut.SetValueRange(0, 1)
    bwLut.Build()  # effective built

    # Now create a lookup table that consists of the full hue circle
    # (from HSV).
    hueLut = vtkLookupTable()
    hueLut.SetTableRange(0, 2000)
    hueLut.SetHueRange(0, 1)
    hueLut.SetSaturationRange(1, 1)
    hueLut.SetValueRange(1, 1)
    hueLut.Build()  # effective built

    # Finally, create a lookup table with a single hue but having a range
    # in the saturation of the hue.
    satLut = vtkLookupTable()
    satLut.SetTableRange(0, 2000)
    satLut.SetHueRange(.6, .6)
    satLut.SetSaturationRange(0, 1)
    satLut.SetValueRange(1, 1)
    satLut.Build()  # effective built

    # Create the first of the three planes. The filter vtkImageMapToColors
    # maps the data through the corresponding lookup table created above.  The
    # vtkImageActor is a type of vtkProp and conveniently displays an image on
    # a single quadrilateral plane. It does this using texture mapping and as
    # a result is quite fast. (Note: the input image has to be unsigned char
    # values, which the vtkImageMapToColors produces.) Note also that by
    # specifying the DisplayExtent, the pipeline requests data of this extent
    # and the vtkImageMapToColors only processes a slice of data.
    sagittalColors = vtkImageMapToColors()
    sagittalColors.SetInputConnection(reader.GetOutputPort())
    sagittalColors.SetLookupTable(bwLut)
    sagittalColors.Update()

    sagittal = vtkImageActor()
    sagittal.GetMapper().SetInputConnection(sagittalColors.GetOutputPort())
    sagittal.SetDisplayExtent(128, 128, 0, 255, 0, 92)

    # Create the second (axial) plane of the three planes. We use the
    # same approach as before except that the extent differs.
    axialColors = vtkImageMapToColors()
    axialColors.SetInputConnection(reader.GetOutputPort())
    axialColors.SetLookupTable(hueLut)
    axialColors.Update()

    axial = vtkImageActor()
    axial.GetMapper().SetInputConnection(axialColors.GetOutputPort())
    axial.SetDisplayExtent(0, 255, 0, 255, 46, 46)

    # Create the third (coronal) plane of the three planes. We use
    # the same approach as before except that the extent differs.
    coronalColors = vtkImageMapToColors()
    coronalColors.SetInputConnection(reader.GetOutputPort())
    coronalColors.SetLookupTable(satLut)
    coronalColors.Update()

    coronal = vtkImageActor()
    coronal.GetMapper().SetInputConnection(coronalColors.GetOutputPort())
    coronal.SetDisplayExtent(0, 255, 128, 128, 0, 92)

    # It is convenient to create an initial view of the data. The
    # FocalPoint and Position form a vector direction. Later on
    # (ResetCamera() method) this vector is used to position the camera
    # to look at the data in this direction.
    aCamera = vtkCamera()
    aCamera.SetViewUp(0, 0, -1)
    aCamera.SetPosition(0, -1, 0)
    aCamera.SetFocalPoint(0, 0, 0)
    aCamera.ComputeViewPlaneNormal()
    aCamera.Azimuth(30.0)
    aCamera.Elevation(30.0)

    # Actors are added to the renderer.
    renderer.AddActor(sagittal)
    renderer.AddActor(axial)
    renderer.AddActor(coronal)
    renderer.AddActor(skin)
    renderer.AddActor(bone)

    # Turn off bone for this example.
    bone.VisibilityOff()

    # Set skin to semi-transparent.
    skin.GetProperty().SetOpacity(0.5)

    # An initial camera view is created.  The Dolly() method moves
    # the camera towards the FocalPoint, thereby enlarging the image.
    renderer.SetActiveCamera(aCamera)

    # Calling Render() directly on a vtkRenderer is strictly forbidden.
    # Only calling Render() on the vtkRenderWindow is a valid call.
    renWin.Render()

    renderer.ResetCamera()
    aCamera.Dolly(1.5)

    # Note that when camera movement occurs (as it does in the Dolly()
    # method), the clipping planes often need adjusting. Clipping planes
    # consist of two planes: near and far along the view direction. The
    # near plane clips out objects in front of the plane; the far plane
    # clips out objects behind the plane. This way only what is drawn
    # between the planes is actually rendered.
    renderer.ResetCameraClippingRange()

    # Interact with the data.
    renWin.Render()
    interactor.Initialize()
    interactor.Start()


if __name__ == '__main__':
    main()