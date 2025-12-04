import cv2
import sys

def show(filename):
    src = cv2.imread(filename)  # 画像を読み込み、配列にする

    # 画像が正常にオープンできたか確認
    if src is None:
        print("Not load image")
        sys.exit(-1)  # Exit with success

    print("画像全体")
    print(src)

    print("画像の一部分のピクセルを見る")
    pixel = src[50, 100]  # (y, x)
    print("Pixel value (BGR):", pixel)

    print("画像のピクセル内の各色を見る")
    subpixel = src[50, 100, 0]  # (y, x)[0]
    print("Sub pixel value (B):", subpixel)
