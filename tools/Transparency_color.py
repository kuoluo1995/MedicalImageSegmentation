import vtk
from vtkmodules.vtkCommonColorPython import vtkNamedColors
from vtkmodules.vtkCommonCorePython import vtkLookupTable
from vtkmodules.vtkFiltersCorePython import vtkContourFilter, vtkProbeFilter
from vtkmodules.vtkIOImagePython import vtkNIFTIImageReader
from vtkmodules.vtkImagingCorePython import vtkImageMapToColors
from vtkmodules.vtkInteractionStylePython import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCorePython import vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor, \
    vtkPolyDataMapper, vtkActor, vtkImageActor

colors = vtkNamedColors()
colors.SetColor("SkinColor", 255, 125, 64, 255)
colors.SetColor("BkgColor", 51, 77, 102, 255)

def create_nii_model(file_name, number, color):
    reader = vtkNIFTIImageReader()
    reader.SetFileName(file_name)
    reader.TimeAsVectorOn()
    reader.Update()

    surface = vtkContourFilter()
    surface.SetInputConnection(reader.GetOutputPort())
    surface.SetValue(0, 1)

    lut = vtkLookupTable()
    lut.SetNumberOfColors(number)
    lut.SetTableValue(*color)
    # lut.SetSaturationRange(1.0, 1.0)
    # lut.SetHueRange(0.0, 1.0)
    lut.Build()

    mapToC = vtkImageMapToColors()
    mapToC.PassAlphaToOutputOn()
    mapToC.SetLookupTable(lut)
    mapToC.SetInputConnection(reader.GetOutputPort())
    mapToC.Update()

    probe = vtkProbeFilter()
    probe.SetInputConnection(surface.GetOutputPort())
    probe.SetSourceConnection(mapToC.GetOutputPort())
    probe.Update()

    data_mapper = vtkPolyDataMapper()
    data_mapper.SetInputConnection(probe.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(data_mapper)
    return actor

image_file = 'E:/Corpus_Callosum/Achen_caixing/image.nii'
label_file = 'E:/Corpus_Callosum/Achen_caixing/label.nii'

ren = vtkRenderer()
ren.SetBackground(0, 0, 0)
renWin = vtkRenderWindow()
renWin.SetSize(800, 600)
renWin.AddRenderer(ren)
iren = vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
image = create_nii_model(image_file, 8, (7, 0.9608, 0.8706, 0.7020, 0.5))
label = create_nii_model(label_file, 1, (0, 1, 0, 0, 1))

ren.AddActor(image)
ren.AddActor(label)

iren.Initialize()
renWin.Render()
iren.Start()
