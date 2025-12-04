import cv2
import sys
from IPython.display import Image, display, update_display

def imshow(img, name):
    '''画像を Notebook 上にインラインで表示する。'''
    img = cv2.imencode('.png', img)[1]
    update_display(Image(img), display_id=name)

def named_show(name):
    '''画像を Notebook 上にインラインで表示する。'''
    display(None, display_id=name)

def show(filename):
    src = cv2.imread(filename)  # 画像を読み込み、配列にする

    # 画像が正常にオープンできたか確認
    if src is None:
        print("Not load image")
        sys.exit(-1)  # Exit with success

    roi = src[150:300, 200:350]

    named_show("roi")      # 画像を表示する枠を作成（授業用自作関数）
    imshow(roi, "roi")     # 画像を表示（授業用自作関数）
