import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

def image_preprocessing(path, desired_size):
    i=0
    dir = os.listdir(path)
    for image in dir:
        i=i+1
        img = cv2.imread(os.path.join(path, image), cv2.IMREAD_UNCHANGED)
        desired_size = (128, 128)
        # resize image
        resized = cv2.resize(img, desired_size, interpolation=cv2.INTER_AREA)
        os.remove(os.path.join(path, image))
        print("Image succesfully erased")
        cv2.imwrite(os.path.join(path, "imagen{}.jpg".format(i)), resized)
        print("Image succesfully written")




