import cv2
import sys
from IPython.display import Image, display, update_display
import time

def imshow(img, name):


  img = cv2.imencode('.png', img)[1]
  update_display(Image(img), display_id=name)

def named_show(name):


  display(None, display_id=name)

def show(video_path):
    cap = cv2.VideoCapture(video_path)


    if not cap.isOpened():
        print("動画ファイルを開けませんでした。")
        sys.exit(-1)
    else:
        print("動画ファイルが正常にオープンされました。")

    named_show("test")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        imshow(frame, "test")

        time.sleep(0.01)

    cap.release()                   
