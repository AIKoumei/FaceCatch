import cv2
import sys
import os.path
import numpy as np

def detect(filename, cascade_file = "FaceDetect/lbpcascade_animeface.xml"):
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
        return
    print("detect faces: {}".format(faces.shape[0]))
                                         
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # cv2.imshow("AnimeFaceDetect", image)
    cv2.waitKey(0)
    cv2.imwrite("FaceDetectResult/face_detect.png", image)
    # save nparray as file data
    np.savetxt("FaceDetectResult/face_detect.faces", np.array(faces), fmt="%d")

if len(sys.argv) != 2:
    sys.stderr.write("error: require file name\n")
    sys.stderr.write("usage: detect.py <filename>\n")
    sys.exit(-1)
    
detect(sys.argv[1])
