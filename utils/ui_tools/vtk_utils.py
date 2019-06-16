import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkCommonCorePython import vtkLookupTable
from vtkmodules.vtkFiltersCorePython import vtkFlyingEdges3D, vtkDecimatePro, vtkSmoothPolyDataFilter, \
    vtkPolyDataNormals, vtkContourFilter, vtkProbeFilter
from vtkmodules.vtkIOImagePython import vtkNIFTIImageReader
from vtkmodules.vtkImagingCorePython import vtkImageMapToColors
from vtkmodules.vtkInteractionStylePython import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCorePython import vtkRenderer, vtkPolyDataMapper, vtkProperty, vtkActor

from utils.ui_tools.config import ErrorObserver

error_observer = ErrorObserver()


def create_renderer():
    renderer = vtkRenderer()
    vtk_widget = QVTKRenderWindowInteractor()
    renderer_window = vtk_widget.GetRenderWindow()
    interactor = renderer_window.GetInteractor()

    renderer_window.AddRenderer(renderer)
    interactor.SetRenderWindow(renderer_window)
    interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
    return renderer, vtk_widget, interactor, renderer_window


def create_extractor(reader):
    extractor = vtkFlyingEdges3D()
    extractor.SetInputConnection(reader.GetOutputPort())
    return extractor


def create_polygon_renderer(extractor):
    renderer = vtkDecimatePro()
    renderer.AddObserver('ErrorEvent', error_observer)
    renderer.SetInputConnection(extractor.GetOutputPort())
    renderer.SetTargetReduction(0.5)
    renderer.PreserveTopologyOn()
    return renderer


def create_smoother(renderer, smoothness):
    smoother = vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(renderer.GetOutputPort())
    smoother.SetNumberOfIterations(smoothness)
    return smoother


def create_normals(smoother):
    normals = vtkPolyDataNormals()
    normals.SetInputConnection(smoother.GetOutputPort())
    normals.SetFeatureAngle(60.0)
    return normals


def create_mapper(normals):
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    mapper.ScalarVisibilityOff()
    mapper.Update()
    return mapper


def create_property(color, opacity):
    property = vtkProperty()
    property.SetColor(color[0], color[1], color[2])
    property.SetOpacity(opacity)
    return property


def create_actor(mapper, property):
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.SetProperty(property)


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
