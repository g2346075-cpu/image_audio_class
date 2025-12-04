import os
os.chdir(f"/content/drive/MyDrive/image_audio_class")

import IPython
import wave
import numpy as np

def play(wav_filename):
    wav_file = wave.open(wav_filename, "r")

    print("オーディオチャンネル数", wav_file.getnchannels())
    # オーディオチャンネル数（モノラルなら 1 、ステレオなら 2）

    print("サンプリングレート", wav_file.getframerate(), "Hz")
    # サンプリングレート、普通のCDは44100

    print("bytes( 1 bytes = 8 bit)", wav_file.getsampwidth(), "bytes",
          wav_file.getsampwidth() * 8, "bit")
    # 1サンプルあたりのバイト数、2なら2bytes(16bit)、3なら24bit

    print("データ長", wav_file.getnframes(), " ",
          wav_file.getnframes() / wav_file.getframerate(), "sec")
    # データの個数　サンプリングレートとの対応関係から　秒数

    buf = wav_file.readframes(wav_file.getnframes())
    data = np.frombuffer(buf, dtype="int16")

    return IPython.display.Audio(data, rate=wav_file.getframerate())
