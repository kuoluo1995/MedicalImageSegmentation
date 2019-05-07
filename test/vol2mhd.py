import os
from pathlib import Path

image_data = list(Path('E:\source_NF').rglob('STIR.vol'))
for image in image_data:
    image = str(image).replace('stir', 'STIR')
    path = Path(image).parent
    os.system('E:/source_NF/VolConverter.exe 3D I ' + str(image) + ' ' + str(path) + '/STIR')
    label = str(image).replace('STIR.vol', 'STIR-label.vol')
    os.system('E:/source_NF/VolConverter.exe 3D M ' + str(label) + ' ' + str(path) + '/STIR_label')
