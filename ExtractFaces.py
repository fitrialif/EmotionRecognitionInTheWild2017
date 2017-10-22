'''
This Script is for extracting every face in the image dataset.
Run the script for each subdirectory of dataset
'''

import dlib
from skimage import io
import os
from skimage.transform import resize

path = './DATA/Val/Negative/'
targetPath = './DATA2/Val/Negative/'

if not os.path.exists(targetPath):
    os.makedirs(targetPath)

dirList = os.listdir(path)

detector = dlib.get_frontal_face_detector()

for index,f in enumerate(dirList):
    print("Processing file: {}".format(f))
    try:
        img = io.imread(path+f)
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        for i, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                i, d.left(), d.top(), d.right(), d.bottom()))
            face = img[d.top():d.bottom(),d.left():d.right(),:]
            face = resize(face,(224,224))
            io.imsave(targetPath+str(index)+'_'+str(i)+'.jpeg',face)
    except Exception as e:
        print e
        
