import sys
import os.path
import numpy as np
from PIL import Image 

def SpliteFace(filename = "FaceDetectResult/face_detect.png"):
    image = Image.open(filename)
    size = {'x': 128, 'y': 128}

    # read face pos as [[x, y, w, h],...]
    result = np.loadtxt("FaceDetectResult/face_detect.faces", dtype=int)

    # write face
    for i, info in enumerate(result):
        x = int(info[0] + info[3]*0.5 - size['x']*0.5)
        y = int(info[1] + info[2]*0.5 - size['y']*0.5)
        face = image.crop((x, y, x + size['x'], y + size['y']))
        face.save("FaceSpliteResult/{}.png".format(i), "PNG")

    # cv2.imwrite("Result/face_detect.png", image)

SpliteFace()
