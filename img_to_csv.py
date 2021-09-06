from PIL import Image
import numpy as np
import sys
import os
import csv

data_img = []
# default format can be changed as needed
def createFileList(myDir, format='.png'):
    fileList = []
    #print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

# load the original image
myFileList = createFileList('D:\Final Year Project\CARLA_0.9.5\PythonAPI\examples\_out')#'/path_to_directory_with_images/')





data_img = myFileList
np.savetxt('data_img.txt', data_img, fmt='%s')
print('done!')