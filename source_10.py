import cv2
import sys
import time
import numpy as numpy
import matplotlib.pyplot as plt
from IPython.display import display

def show(filename):
    src = cv2.imread(filename)


    if src is None:
        print("Not load image")
        sys.exit(-1) #Exit with success

    roi = src[150:300, 200:350]

    fig = plt.figure(figsize=(8,6))
    display_handle = display(fig, display_id=True)
    for i in range(0, 100, 1):
        b = roi[:, i+1, 0]

        plt.clf()
        plt.plot(b)
        plt.ylim(0, 255)
        plt.xlim(0, 150)
        display_handle.update(fig)
        time.sleep(0.5)    