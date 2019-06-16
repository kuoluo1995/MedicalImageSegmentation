import sys

from PyQt5.QtWidgets import QApplication
from utils.ui_tools.main_windows import MainWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow(app)
    sys.exit(app.exec_())
