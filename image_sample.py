import cv2
import sys
from google. colab. patches import cv2_imshow
from IPython. display import Image, display, update_display
import time
import numpy as np
import matplotlib. pyplot as plt

#本来であれば、cv2.imshow(srcMat)やcv2_imshow(srcMat)を利用するが、
#Colabotaroyだと不便のため、授業用自作関数によって対応。
#ほかの手法に変更することもOK。

def imshow(img, name):

  img = cv2. imencode('.png', img) [1]
  update_display(Image(img), display_id=name)

def named_show(name):

  display(None, display_id=name)

def show_image(file_path, is_color=1, window_name="src"):
  srcMat= cv2.imread(file_path,is_color)#画像を読み込み、配列にする

  # 画像が正常にオープンできたか確認
  if srcMat is None:
    print("Not load image:画像がロードできません。ファイル名を確認してください。")
    sys. exit(-1) # Exit with success

  srcMat = cv2. resize(srcMat, (500,500))

  named_show(window_name) #画像を表示する枠を作成(授業用自作関数)
  imshow(srcMat,window_name)#画像を表示(授業用自作関数)

  #配列の次元によって、場合分けをする。
  if len(srcMat.shape) == 2:#チャンネル方向への次元がないため、2次元
    height = srcMat. shape[0]
    width = srcMat. shape[1]
    #height, width = srcMat. shape[:2] でも可

    print(f"画像横幅:{width},画像縱幅:{height}")
    print(f"画像色チャンネル:1")
  else:#チャンネル方向への次元がないため、カラー画像は3次元
    height = srcMat. shape[0]
    width = srcMat. shape[1]
    channels=srcMat.shape[2]##3次元の場合取得可能。
    #height, width, channel = srcMat. shape[:3] でも可

    print(f"画像横幅 : {width},画像縱幅 : {height}")
    print(f"画像色チャンネル　3はカラー、1はグレースケール : {channels}")

def pickcolor_image(file_path, lower_hue, upper_hue, window_name="src"):
  srcMat =cv2.imread(file_path,1)#画像を読み込み、配列にする

  # 画像が正常にオープンできたか確認
  if srcMat is None:
    print("Not load image : 画像がロードできません。ファイル名を確認してください。")
    sys. exit(-1) # Exit with success

  named_show(window_name)#画像を表示する枠を作成(授業用自作関数)
  imshow(srcMat,window_name)#画像を表示(授業用自作関数)

  hsv_image = cv2.cvtColor(srcMat,cv2.COLOR_BGR2HSV)#画像をRGB空間から、HSV色空間へ変換する

  lower_y = np.array([15, 0, 0])
  upper_y = np.array([35, 255, 255])

  lower_g = np.array([50, 0, 0])
  upper_g = np.array([85, 255, 255])


  # 4. 指定した範囲内の色を抽出するマスクを作成
  mask_y = cv2.inRange(hsv_image, lower_y, upper_y)
  mask_g = cv2.inRange(hsv_image, lower_g, upper_g)

  mask = cv2.bitwise_or(mask_y, mask_g)
  

  # 5. 元画像とマスクを重ね合わせて色を抽出
  srcMat = cv2.bitwise_and(srcMat, srcMat, mask=mask)

  named_show(window_name)#画像を表示する枠を作成(授業用自作関数)
  imshow(srcMat,window_name)#画像を表示(授業用自作関数)  

def show_histgram_hsv(file_path, window_name="src"):
    srcMat = cv2.imread(file_path)
    if srcMat is None:
        print("Not load image")
        sys.exit(-1)

    hsvMat = cv2.cvtColor(srcMat, cv2.COLOR_BGR2HSV)

    channels = ('H', 'S', 'V')
    colors = ('m', 'c', 'k')  # 見やすさ用

    plt.figure(figsize=(10, 5))
    plt.title('HSV Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    for i, ch in enumerate(channels):
        if i == 0:
            hist = cv2.calcHist([hsvMat], [i], None, [180], [0, 180])
        else:
            hist = cv2.calcHist([hsvMat], [i], None, [256], [0, 256])
        plt.plot(hist, label=ch)

    plt.legend()
    plt.show()

def analyze_hsv_statistics(file_path):
    import cv2
    import numpy as np
    import sys

    srcMat = cv2.imread(file_path)
    if srcMat is None:
        print("Not load image")
        sys.exit(-1)

    hsvMat = cv2.cvtColor(srcMat, cv2.COLOR_BGR2HSV)

    stats = {}
    channels = {'H': 0, 'S': 1, 'V': 2}

    for name, idx in channels.items():
        channel_data = hsvMat[:, :, idx]
        stats[name] = {
            'mean': np.mean(channel_data),
            'std': np.std(channel_data),
            'min': np.min(channel_data),
            'max': np.max(channel_data)
        }

    return stats


def histogram_equal(file_path, window_name="src"):
  srcMat= cv2.imread(file_path)#画像を読み込み、配列にする

  #画像が正常にオープンできたか確認
  if srcMat is None:
    print("Not load image:画像がロードできません。ファイル名を確認してください。")
    sys. exit(-1) # Exit with success

  named_show(window_name) #画像を表示する枠を作成(授業用自作関数)
  imshow(srcMat,window_name)#画像を表示(授業用自作関数)

  gray_image = cv2.cvtColor(srcMat,cv2.COLOR_BGR2GRAY)#画像をRGB空間から、HSV色空間へ変換する

  # ヒストグラム平坦化の実行
  equ = cv2. equalizeHist(gray_image)

  # 結果の横並び表示
  res = np.hstack((gray_image, equ))

  named_show(window_name+"equ") #画像を表示する枠を作成(授業用自作関数)
  imshow(res,window_name+"equ")#画像を表示(授業用自作関数)

  # グラフの設定
  images = (gray_image, equ)
  plt. figure(figsize=(10, 5))
  plt. title('Color Histogram' )
  plt. xlabel('Pixel Value (0-255)')
  plt. ylabel (' Frequency' )

  labels = ["Gray", "EQU"]
  # 各チャンネルごとにヒストグラムを計算して描画
  for i, img in enumerate(images):
      # cv2.calcHist([画像],[チャンネル],マスク,[BIN数],[範囲])
      hist = cv2.calcHist([img], [0], None, [256], [0, 256])
      plt. plot(hist, label=labels[i])
      plt.xlim([0, 256])

  plt. legend()
  plt. show()

def Hough_image(file_path, window_name="src"):
  srcMat =cv2.imread(file_path)#画像を読み込み、配列にする

  #画像が正常にオープンできたか確認
  if srcMat is None:
    print("Not load image:画像がロードできません。ファイル名を確認してください。")
    sys. exit(-1) # Exit with success

  gray = cv2. cvtColor(srcMat, cv2. COLOR_BGR2GRAY)
  edges = cv2. Canny(gray, 150, 250, apertureSize=3)

  named_show(window_name+"edge") #画像を表示する枠を作成(授業用自作関数)
  imshow(edges,window_name+"edge")#画像を表示(授業用自作関数)

  # 確率的ハフ変換による直線検出
  # rho:距離の解像度,theta:角度の解像度,threshold:直線とみなす最低限の投票数
  # minLineLength:直線とみなす最小の長さ,maxLineGap:同一線とみなす最大の間隔
  lines = cv2. HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80,
                           minLineLength=100, maxLineGap=10)

  if lines is not None:
      for line in lines:
          x1, y1, x2, y2 = line[0]
          # 検出した直線を描画
          cv2.line(srcMat, (x1, y1), (x2, y2), (0, 255, 0), 2)

  named_show(window_name)#画像を表示する枠を作成(授業用自作関数)
  imshow(srcMat,window_name)#画像を表示(授業用自作関数)
def Circle_Hough_image(file_path, window_name="src"):
  srcMat=cv2.imread(file_path)#画像を読み込み、配列にする

  # 画像が正常にオープンできたか確認
  if srcMat is None:
    print("Not load image:画像がロードできません。ファイル名を確認してください。")
    sys. exit(-1) # Exit with success

  srcMat = cv2. resize(srcMat, (600, 400))

  gray = cv2. cvtColor(srcMat, cv2. COLOR_BGR2GRAY)
  ret, binary = cv2. threshold(gray, 240, 255, cv2. THRESH_BINARY)
  edges = cv2. Canny(binary, 100, 250, apertureSize=7)

  named_show(window_name+"edge") #画像を表示する枠を作成(授業用自作関数)
  imshow(edges, window_name+"edge")#画像を表示(授業用自作関数)

  # ハフ変換による円検出
  # method: HOUGH_GRADIENT,dp:解像度の比率,minDist:円同士の最小距離
  # paraml:Cannyの上限値,param2:中心投票の閾値(小さいほど多くの円を拾う)
  circles = cv2. HoughCircles(gray, cv2. HOUGH_GRADIENT, dp=1, minDist=20,
                             param1=255, param2=70, minRadius=0, maxRadius=0)
  if circles is not None:
      circles = np.uint16(np.around(circles))#Opencvの描画系は小数点が利用できないため、整数とする。
      for i in circles[0, :]:
          # 外周を描画
          cv2.circle(srcMat, (i[0], i[1]), i[2], (0, 255, 0), 2)
          # 中心を描画
          cv2.circle(srcMat, (i[0], i[1]), 2, (0, 0, 255), 3)

  named_show(window_name) #画像を表示する枠を作成(授業用自作関数)
  imshow(srcMat,window_name)#画像を表示(授業用自作関数)