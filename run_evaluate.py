import os

fold_names = os.listdir('E:/Slimed_NF')
for fold in fold_names:
    files = os.listdir('E:/Slimed_NF/' + fold)
    for file in files:
        if ('label' in file):
            os.system('E:/VolConverter.exe 3D M E:/Slimed_NF/' + fold + '/' + file + ' E:/Slimed_NF/' + fold + '/' +
                      file.split('.')[0])
        else:
            os.system('E:/VolConverter.exe 3D I E:/Slimed_NF/' + fold + '/' + file + ' E:/Slimed_NF/' + fold + '/' +
                      file.split('.')[0])
