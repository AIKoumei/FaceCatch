import cv2
import sys
import os.path
import numpy as np
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
    if isinstance(faces, tuple):
        print("detect faces: 0")
        sys.exit(0)
    print("detect faces: {}".format(faces.shape[0]))
    return faces
                                         
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # cv2.imshow("AnimeFaceDetect", image)
    # cv2.waitKey(0)
    # cv2.imwrite("FaceDetectResult/face_detect.png", image)
    # save nparray as file data
    # np.savetxt("FaceDetectResult/face_detect.faces", np.array(faces), fmt="%d")

def SpliteFace(image, data):
    size = {'x': 128, 'y': 128}

    # write face
    for i, info in enumerate(data):
        x = int(info[0] + info[3]*0.5 - size['x']*0.5)
        y = int(info[1] + info[2]*0.5 - size['y']*0.5)
        face = image.crop((x, y, x + size['x'], y + size['y']))
        face.save("FaceSpliteResult/{}.png".format(i), "PNG")

# run
if len(sys.argv) != 2:
    sys.stderr.write("error: require file name\n")
    sys.stderr.write("usage: main.py <filename>\n")
    sys.exit(-1)

filename = sys.argv[1]
SpliteFace(Image.open(filename), DetectFace(filename))
