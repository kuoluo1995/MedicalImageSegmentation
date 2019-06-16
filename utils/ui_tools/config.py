APPLICATION_TITLE = "3D Visualizer"
N_LABELS = 2

BRAIN_COLORS = [(1.0, 0.9, 0.9)]  # RGB percentages
BRAIN_INTENSITY = 2.0

MASK_COLORS = [(1, 0.5, 0.5),
               (0.5, 1, 0.5),
               (0.5, 0.5, 1)]  # RGB percentages


class ErrorObserver:
    __ErrorOccurred = False
    __ErrorMessage = None
    CallDataType = 'string0'

    def __call__(self, *args, **kwargs):
        self.__ErrorOccurred = True
        self.__ErrorMessage = kwargs['message']
