import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
from tqdm import tqdm
from PIL import Image

cwd = os.getcwd() 
data_path = join(cwd,'QMUL_360degreeViewSphere_FaceDatabase/Set1_Greyscale')

folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
allfiles=[]
for i, folder in enumerate(tqdm(folder_list)):
    print(folder)

    folder_path = join(data_path, folder)
    allfiles.append(
        [join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path, cf)) and cf[0] != '.') and 'ras' in cf])


# First let's rewrite all these RAS images as JPEGS because we like JPGS
new_data_path = join(cwd,'images/')
for i, folder in enumerate(tqdm(folder_list)):
    os.makedirs(join(new_data_path,folder), exist_ok=True)
    for file in allfiles[i]:
      theim = Image.open(file)
      newim = theim.convert('RGB')
      newim.save(join(new_data_path,folder,file.split('/')[-1].replace('ras','jpg')))


# Here we go again
data_path = join(cwd,'images/')

folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(folder_list,range(0,len(folder_list))))


savedir = cwd
dataset_list = ['base','val','novel']

folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(folder_list,range(0,len(folder_list))))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list_all.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])
    random.shuffle(classfile_list_all[i])


for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        if 'base' in dataset:
            if (i%2 == 0):
                file_list = file_list + classfile_list
                label_list = label_list + [int(v.split('_')[-1].replace('.jpg','')) for v in classfile_list]
        if 'val' in dataset:
            if (i%4 == 1):
                file_list = file_list + classfile_list
                label_list = label_list + [int(v.split('_')[-1].replace('.jpg','')) for v in classfile_list]
        if 'novel' in dataset:
            if (i%4 == 3):
                file_list = file_list + classfile_list
                label_list = label_list + [int(v.split('_')[-1].replace('.jpg','')) for v in classfile_list]

    fo = open(dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folder_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in file_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in label_list])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)

