import platform
import sys
import time
import os.path

import shutil

import numpy as np
import cv2
from PIL import Image 


def DetectFace(filename, cascade_file = "FaceDetect/lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))

    # show face length
    count = 0
    if isinstance(faces, tuple):
        print("detect faces: 0")
    else:
        count = faces.shape[0]
        print("detect faces: {}".format(count))
    return faces, count

size = (128, 128)
def SplitFace(file, data):
    # if no face detected
    if isinstance(data, tuple):
        return
        
    image = Image.open(file)

    # write face
    root = "FaceSplitResult"
    if data.shape[0] > 1:
        root = '/'.join(file[:file.rfind(SEPARATOR)].split(SEPARATOR))
        CheckFolder(root)
    for i, info in enumerate(data):
        w = info[2]*1.5
        h = info[3]*1.5
        x = int(info[0] + info[3]*0.5 - h*0.5)
        y = int(info[1] + info[2]*0.5 - w*0.5)
        face = image.crop((x, y, x + w, y + h)).resize(size)
        # sub file name
        subname = file[:file.rfind('.')]
        arr = subname.split(SEPARATOR)
        face.save((len(arr) > 2 and "{}/{}/{}.png" or "{}/{}-{}.png").format(root, arr[-1], i), "PNG")

# ######################################################################
#   file copy
# ######################################################################
def MoveFile(srcfile,dstfile):
    fpath, fname = os.path.split(dstfile)    #分离文件名和路径
    if not os.path.exists(fpath):
        os.makedirs(fpath)                #创建路径
    shutil.move(srcfile, dstfile)          #移动文件

def CopyFile(srcfile,dstfile):
    fpath, fname = os.path.split(dstfile)    #分离文件名和路径
    if not os.path.exists(fpath):
        os.makedirs(fpath)                #创建路径
    shutil.copyfile(srcfile, dstfile)      #复制文件
# ######################################################################
IMAGE_EXT = (
    '.png', '.jpg'
)

def ScanFolder(path):
    global CUR_IMAGE
    list = os.listdir(path)
    for i in range(0, len(list)):
        file = os.path.join(path, list[i])
        if os.path.isfile(file):
            print("scan file: {}".format(file))
            #  check if is image
            is_image = False
            for ext in IMAGE_EXT:
                if file.endswith(ext):
                    is_image = True
                    break
            # if not image then continue
            if not is_image:
                print("skip file: {}".format(file))
                continue
            # detect image
            face, count = DetectFace(file)
            # if count == 0:
            #     CopyFile(file, "FaceDetectFailed/{}".format(file.split(IS_WINDOWS and '\\' or IS_LINUX and '/' or "")[-1]))
            # else:
            #     SplitFace(file, face)
            SplitFace(file, face)
            # plus count
            CUR_IMAGE += 1
            print("Total {}/{}({:.2f}%), scanning: {}".format(CUR_IMAGE, TOTAL_IMAGE, CUR_IMAGE*TOTAL_PERCENT*100, file))
        else:
            ScanFolder(path)

            
def GetImageFileCounts(path):
    folders = [path]
    count = 0
    folder = folders.pop(0)
    files = os.listdir(folder)
    while(folder):
        files = os.listdir(folder)
        for f in files:
            file = os.path.join(path, f)
            # check file
            if os.path.isfile(file):
                #  check if is image
                is_image = False
                for ext in IMAGE_EXT:
                    if file.endswith(ext):
                        is_image = True
                        break
                # if not image then continue
                if not is_image:
                    # print("skip file: {}".format(file))
                    continue
                count += 1
            else:
                folders.append(file)
        folder = len(folders) > 0 and folders.pop(0) or False
    return count

def CheckFolder(path):
    if not os.path.isdir(path):
        os.makedirs(path)
# ######################################################################
# params
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

SEPARATOR = IS_WINDOWS and "\\" or IS_LINUX and "/" or ""

# check folders
check = (
    'ImageForDetect',
    'FaceDetectResult',
    'FaceDetectFailed',
    'FaceSplitResult',
)
for folder in check:
    CheckFolder(folder)

# run
TOTAL_IMAGE = GetImageFileCounts("ImageForDetect")
TOTAL_PERCENT = TOTAL_IMAGE > 0 and 1/TOTAL_IMAGE or 1
CUR_IMAGE = 0

# 
start_time = time.time()
ScanFolder("ImageForDetect")
finished_time = time.time() - start_time
print("finished {} sec".format(finished_time))
