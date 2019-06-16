import vtk
from vtkmodules.vtkRenderingCorePython import vtkImageProperty, vtkImageSlice
from vtkmodules.vtkRenderingImagePython import vtkImageResliceMapper

from utils.ui_tools.config import N_LABELS
from utils.ui_tools.vtk_utils import create_polygon_renderer, create_extractor, create_smoother, create_normals, \
    create_mapper, create_property, create_actor, create_nii_model


class NiiLabel:
    color = None
    opacity = None
    smoothness = None

    _extractor = None
    actor = None
    property = None
    smoother = None

    def set_config(self, color, opacity, smoothness, nii_object):
        self.color = color
        self.opacity = opacity
        self.smoothness = smoothness
        self._extractor = create_extractor(nii_object.reader)

    def add_surface_rendering(self, label_value):
        self._extractor.SetValue(0, label_value)
        self._extractor.Update()

        if self._extractor.GetOutput().GetMaxCellSize():
            renderer = create_polygon_renderer(self._extractor)
            self.smoother = create_smoother(renderer, self.smoothness)
            normals = create_normals(self.smoother)
            actor_mapper = create_mapper(normals)
            self.property = create_property(self.color, self.opacity)
            self.actor = create_actor(actor_mapper, self.property)


class NiiObject:
    file_path = None
    reader = None
    extent = None
    labels = []
    scalar_range = None
    image_mapper = None
    image_property = None
    axial = None
    coronal = None
    sagittal = None
    view_array = None

    def read_file(self, renderer, file_path, color, is_mask):
        def _read_volume():
            reader = vtk.vtkNIFTIImageReader()
            reader.SetFileNameSliceOffset(1)
            reader.SetDataByteOrderToBigEndian()
            reader.SetFileName(file_path)
            reader.Update()
            return reader

        self.file_path = file_path
        self.reader = _read_volume()
        self.extent = self.reader.GetDataExtent()

        if not is_mask:
            n_labels = 1
            self.scalar_range = self.reader.GetOutput().GetScalarRange()
            color_table = vtk.vtkLookupTable()
            color_table.SetTableRange(self.scalar_range)
            color_table.SetSaturationRange(0, 0)
            color_table.SetHueRange(0, 0)
            color_table.SetValueRange(0, 2)
            color_table.Build()
            self.image_mapper = vtk.vtkImageMapToColors()
            self.image_mapper.SetInputConnection(self.reader.GetOutputPort())
            self.image_mapper.SetLookupTable(color_table)
            self.image_mapper.Update()
        else:
            n_labels = int(self.reader.GetOutput().GetScalarRange()[1])
            n_labels = n_labels if n_labels <= N_LABELS else N_LABELS
        for i in range(n_labels):
            label = NiiLabel()
            label.set_config(color[i], 1.0, 1000, self)
            if not is_mask:
                label.add_surface_rendering(sum(self.scalar_range) / 2)
            else:
                label.add_surface_rendering(i + 1)
            renderer.AddActor(label.actor)
            self.labels.append(label)

    def init_property(self, renderer):
        action = create_nii_model(self.file_path,8, (7, 0.9608, 0.8706, 0.7020, 0.5))
        renderer.AddActor(action)

    def init_slicer(self, renderer):
        x = self.extent[1]
        y = self.extent[3]
        z = self.extent[5]

        self.axial = vtk.vtkImageActor()
        axial_prop = vtk.vtkImageProperty()
        axial_prop.SetOpacity(0)
        self.axial.SetProperty(axial_prop)
        self.axial.GetMapper().SetInputConnection(self.image_mapper.GetOutputPort())
        self.axial.SetDisplayExtent((0, x, 0, y, int(z / 2), int(z / 2)))
        self.axial.InterpolateOn()
        self.axial.ForceOpaqueOn()

        self.coronal = vtk.vtkImageActor()
        cor_prop = vtk.vtkImageProperty()
        cor_prop.SetOpacity(0)
        self.coronal.SetProperty(cor_prop)
        self.coronal.GetMapper().SetInputConnection(self.image_mapper.GetOutputPort())
        self.coronal.SetDisplayExtent((0, x, int(y / 2), int(y / 2), 0, z))
        self.coronal.InterpolateOn()
        self.coronal.ForceOpaqueOn()

        self.sagittal = vtk.vtkImageActor()
        sag_prop = vtk.vtkImageProperty()
        sag_prop.SetOpacity(0)
        self.sagittal.SetProperty(sag_prop)
        self.sagittal.GetMapper().SetInputConnection(self.image_mapper.GetOutputPort())
        self.sagittal.SetDisplayExtent((int(x / 2), int(x / 2), 0, y, 0, z))
        self.sagittal.InterpolateOn()
        self.sagittal.ForceOpaqueOn()

        renderer.AddActor(self.axial)
        renderer.AddActor(self.coronal)
        renderer.AddActor(self.sagittal)
        self.view_array = (self.axial, self.coronal, self.sagittal)
