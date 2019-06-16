import math

from PyQt5.QtWidgets import QGridLayout, QLabel, QFrame, QMainWindow, QApplication

from utils.ui_tools import vtk_utils
from utils.ui_tools.config import BRAIN_COLORS, MASK_COLORS, APPLICATION_TITLE, BRAIN_INTENSITY
from utils.ui_tools.object_utils import NiiObject
from utils.ui_tools.qt_ui_utils import *

BRAIN_FILE = 'E:/Corpus_Callosum/Achen_caixing/image.nii'
MASK_FILE = 'E:/Corpus_Callosum/Achen_caixing/label.nii'


class MainWindow(QMainWindow, QApplication):
    renderer = None
    vtk_widget = None
    interactor = None
    render_window = None

    slice_dict = None

    brain = NiiObject()
    brain_projection_check_box = None
    brain_slicer_check_box = None
    brain_intensity_spin_box = None

    mask = NiiObject()

    def __init__(self, app):
        self.app = app
        QMainWindow.__init__(self, None)
        self.renderer, self.vtk_widget, self.interactor, self.render_window = vtk_utils.create_renderer()
        print('renderer完成')

        self.brain.read_file(self.renderer, BRAIN_FILE, BRAIN_COLORS, False)
        self.brain.init_property(self.renderer)
        self.brain.init_slicer(self.renderer)
        print('brain model read完成')
        self.mask.read_file(self.renderer, MASK_FILE, MASK_COLORS, True)
        print('mask model read完成')

        grid = QGridLayout()
        grid.setColumnMinimumWidth(2, 700)
        grid.addWidget(create_vtk_widget(self.brain, self.mask, self.vtk_widget), 0, 2, 5, 5)
        print('vtk widget完成')
        grid.addWidget(self.create_brain_setting_widget(), 0, 0, 1, 2)
        print('brain settingt完成')
        grid.addWidget(self.create_mask_setting_widget(), 1, 0, 1, 2)
        print('mask settingt完成')
        grid.addWidget(self.create_views_widget(), 3, 0, 2, 2)
        print('views widget完成')
        frame = QFrame()
        frame.setAutoFillBackground(True)
        frame.setLayout(grid)
        self.setCentralWidget(frame)

        self.set_axial_view()
        print('axial view完成')
        self.interactor.Initialize()
        self.setWindowTitle(APPLICATION_TITLE)
        self.show()
        print('展示完成')

    def create_brain_setting_widget(self):
        brain_group_box = QGroupBox('Brain Settings')

        row = 0
        brain_group_layout = QGridLayout()

        brain_group_layout.addWidget(QLabel('Intensity'), row, 0)
        self.brain_intensity_spin_box = create_spin_box(3.0, 0.0, 0.1, BRAIN_INTENSITY)
        self.brain_intensity_spin_box.valueChanged.connect(self.brain_intensity_vc)
        brain_group_layout.addWidget(self.brain_intensity_spin_box, row, 1)
        row += 1

        self.brain_projection_check_box = create_check_box('Projection')
        self.brain_projection_check_box.clicked.connect(self.brain_projection_vc)
        brain_group_layout.addWidget(self.brain_projection_check_box, row, 0)
        self.brain_slicer_check_box = create_check_box('Slicer')
        self.brain_slicer_check_box.clicked.connect(self.brain_slicer_vc)
        brain_group_layout.addWidget(self.brain_slicer_check_box, row, 1)
        row += 1

        brain_group_layout.addWidget(create_separator(), row, 0, 1, 3)

        self.slice_dict = {'Axial': {'name': 'Axial Slice', 'change_function': self.axial_slice_changed,
                                     'set_view': self.set_axial_view, 'row': row + 1},
                           'Coronal': {'name': 'Coronal Slice', 'change_function': self.coronal_slice_changed,
                                       'set_view': self.set_coronal_view, 'row': row + 2},
                           'Sagittal': {'name': 'Sagittal Slice', 'change_function': self.sagittal_slice_changed,
                                        'set_view': self.set_sagittal_view, 'row': row + 3}}
        extent_index = 5
        for key, value in self.slice_dict.items():
            slice_slider = create_slider(self.brain.extent[extent_index - 1], self.brain.extent[extent_index],
                                         self.brain.extent[extent_index] / 2)
            brain_group_layout.addWidget(QLabel(value['name']), value['row'], 0)
            brain_group_layout.addWidget(slice_slider, value['row'], 1, 1, 2)
            self.slice_dict[key]['widget'] = slice_slider
            slice_slider.valueChanged.connect(value['change_function'])
            extent_index -= 2

        brain_group_box.setLayout(brain_group_layout)
        return brain_group_box

    def create_mask_setting_widget(self):
        mask_group_box = QGroupBox('Mask Setting')
        mask_group_layout = QGridLayout()
        mask_group_box.setLayout(mask_group_layout)
        return mask_group_box

    def create_views_widget(self):
        views_box = QGroupBox("Views")
        views_box_layout = QVBoxLayout()
        for key, value in self.slice_dict.items():
            view = create_push_button(key)
            view.clicked.connect(value['set_view'])
            views_box_layout.addWidget(view)
        views_box.setLayout(views_box_layout)
        return views_box

    def brain_intensity_vc(self):
        intensity = self.brain.image_mapper.GetLookupTable()
        new_intensity_value = self.brain_intensity_spin_box.value()
        intensity.SetValueRange(0.0, new_intensity_value)
        intensity.Build()
        self.brain.image_mapper.SetLookupTable(intensity)
        self.brain.image_mapper.Update()
        self.render_window.Render()

    def brain_projection_vc(self):
        projection_checked = self.brain_projection_check_box.isChecked()
        self.brain_slicer_check_box.setDisabled(projection_checked)
        self.brain.image_property.SetOpacity(projection_checked)
        self.render_window.Render()

    def brain_slicer_vc(self):
        slicer_checked = self.brain_slicer_check_box.isChecked()
        for key, value in self.slice_dict.items():
            value['widget'].setEnabled(slicer_checked)
        self.brain_projection_check_box.setDisabled(slicer_checked)
        for view in self.brain.view_array:
            view.GetProperty().SetOpacity(slicer_checked)
        self.render_window.Render()

    def set_axial_view(self):
        self.renderer.ResetCamera()
        focal_point = self.renderer.GetActiveCamera().GetFocalPoint()
        position = self.renderer.GetActiveCamera().GetPosition()
        distance = math.sqrt(
            (position[0] - focal_point[0]) ** 2 + (position[1] - focal_point[1]) ** 2 + (
                    position[2] - focal_point[2]) ** 2)
        self.renderer.GetActiveCamera().SetPosition(focal_point[0], focal_point[1], focal_point[2] + distance)
        self.renderer.GetActiveCamera().SetViewUp(0.0, 1.0, 0.0)
        self.renderer.GetActiveCamera().Zoom(1.8)
        self.render_window.Render()

    def axial_slice_changed(self):
        position = self.slice_dict['Axial']['widget'].value()
        self.brain.axial.SetDisplayExtent(self.brain.extent[0], self.brain.extent[1], self.brain.extent[2],
                                          self.brain.extent[3], position, position)
        self.render_window.Render()

    def coronal_slice_changed(self):
        position = self.slice_dict['Coronal']['widget'].value()
        self.brain.coronal.SetDisplayExtent(self.brain.extent[0], self.brain.extent[1], position, position,
                                            self.brain.extent[4], self.brain.extent[5])
        self.render_window.Render()

    def set_coronal_view(self):
        self.renderer.ResetCamera()
        focal_point = self.renderer.GetActiveCamera().GetFocalPoint()
        position = self.renderer.GetActiveCamera().GetPosition()
        distance = math.sqrt(
            (position[0] - focal_point[0]) ** 2 + (position[1] - focal_point[1]) ** 2 + (
                    position[2] - focal_point[2]) ** 2)
        self.renderer.GetActiveCamera().SetPosition(focal_point[0], focal_point[2] - distance, focal_point[1])
        self.renderer.GetActiveCamera().SetViewUp(0.0, 0.5, 0.5)
        self.renderer.GetActiveCamera().Zoom(1.8)
        self.render_window.Render()

    def sagittal_slice_changed(self):
        position = self.slice_dict['Sagittal']['widget'].value()
        self.brain.sagittal.SetDisplayExtent(position, position, self.brain.extent[2], self.brain.extent[3],
                                             self.brain.extent[4], self.brain.extent[5])
        self.render_window.Render()

    def set_sagittal_view(self):
        self.renderer.ResetCamera()
        focal_point = self.renderer.GetActiveCamera().GetFocalPoint()
        position = self.renderer.GetActiveCamera().GetPosition()
        distance = math.sqrt(
            (position[0] - focal_point[0]) ** 2 + (position[1] - focal_point[1]) ** 2 + (
                    position[2] - focal_point[2]) ** 2)
        self.renderer.GetActiveCamera().SetPosition(focal_point[2] + distance, focal_point[0],
                                                    focal_point[1])
        self.renderer.GetActiveCamera().SetViewUp(0.0, 0.0, 1.0)
        self.renderer.GetActiveCamera().Zoom(1.6)
        self.render_window.Render()
