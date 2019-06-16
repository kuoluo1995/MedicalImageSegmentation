import PyQt5.QtCore as Qt
from PyQt5.QtWidgets import QSpinBox, QDoubleSpinBox, QCheckBox, QWidget, QSizePolicy, QSlider, QGroupBox, QVBoxLayout, \
    QPushButton, QRadioButton


def create_spin_box(max_value, min_value, step, picker_value):
    if isinstance(max_value, int):
        picker = QSpinBox()
    else:
        picker = QDoubleSpinBox()
    picker.setMaximum(max_value)
    picker.setMinimum(min_value)
    picker.setSingleStep(step)
    picker.setValue(picker_value)
    return picker


def create_check_box(name):
    check_box = QCheckBox(name)
    return check_box


def create_separator():
    horizontal_line = QWidget()
    horizontal_line.setFixedHeight(1)
    horizontal_line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    horizontal_line.setStyleSheet("background-color: #c8c8c8;")
    return horizontal_line


def create_slider(min_value, max_value, default_value):
    slice_widget = QSlider(Qt.Qt.Horizontal)
    slice_widget.setDisabled(True)
    slice_widget.setRange(min_value, max_value)
    slice_widget.setValue(default_value)
    return slice_widget


def create_vtk_widget(brain, mask, vtk_widget):
    object_title = 'Brain:{0} (min:{1:.2f}, max:{2:.2f} Mask:{3}'.format(brain.file_path,
                                                                         brain.scalar_range[0],
                                                                         brain.scalar_range[1],
                                                                         mask.file_path)
    group_box = QGroupBox(object_title)
    object_layout = QVBoxLayout()
    object_layout.addWidget(vtk_widget)
    group_box.setLayout(object_layout)
    return group_box


def create_push_button(name):
    button = QPushButton(name)
    return button


def create_radio_button(name):
    button = QRadioButton(name)
    return button
