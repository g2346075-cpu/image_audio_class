import numpy as np
import librosa
import librosa.display
import matplotlib. pyplot as plt

def analyze_speech_with_mfccs(file_path, n_mfcc=13):

MFCCsを抽出し、人の抑揚と母音の特徴を分析するためにプロットする。

Args:
    file_path(str):分析したいWAVファイルのパス。
    n_mfcc(int):抽出するMFCCsの数。

try:
    # 1. 音声ファイルの読み込み
    x, sr = librosa. load (file_path, sr=None)

    # 2. MFCCsの計算
    mfccs = librosa. feature. mfcc (y=x, sr=sr, n_mfcc=n_mfcc,
                                    n_fft=2048, hop_length=512)

    # 時間軸の計算
    times = librosa. frames_to_time (np. arange (mfccs. shape[1]), sr=sr, hop_length=512)

    print("¥n --- MFCCs分析情報 --- ")
    print(f"MFCCsの形状(係数数 × フレーム数):{mfccs.shape}")

    # 3. 可視化:スペクトログラムとMFCCsを並列表示
    fig, axes = plt. subplots (3, 1, figsize=(12, 10), sharex=True)

    # --- (A)スペクトログラム（元の周波数情報) ---
    D = librosa. amplitude_to_db (np. abs (librosa. stft(x, n_fft=2048, hop_length=512)), ref=np. max)
    librosa. display. specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[0], cmap=' magma' )
    axes [0]. set(title='Log Power Spectrogram (For Reference)')
    axes [0].set_ylim(0, sr / 2)
    axes [0]. label_outer ()

    # --- (B)MFCCs(メル周波数特徴量)
    img_mfcc = librosa. display. specshow(mfccs, sr=sr, x_axis='time', y_axis=' mel', ax=axes [1], cmap='coolwarm'
    axes [1]. set (title=f'MFCCs (Mel Frequency Feature for Mother Vowel/Timbre Analysis)')
    fig. colorbar (img_mfcc, ax=axes[1], format="%+2. Of dB")
    axes [1]. label_outer ()

    # --- (C)MFCCsの低次係数(抑揚/音の強さの分析)
    # 第0係数:ログエネルギーに相当し、音の強さ(抑揚)を表す
    # 第1係数:声道形状の傾きを表し、母音の基本的な変化(抑揚)に関連

    # 第0係数(エネルギー)を平均化して描画
    axes [2]. plot (times, mfccs [0, :], label=' MFCC 0 (Log Energy)', color='red', linewidth=2)
    axes [2]. set_ylabel (' MFCC 0 Amplitude')

    # 第1係数(基本的な音色)
    ax2_twin = axes[2].twinx() # 軸を共有しない2つ目のY軸を作成
    ax2_twin. plot(times, mfccs [1, :], label='MFCC 1 (Timbre/Slope)', color='blue', linestyle=' -- ', linewidth=1.5)
    ax2 twin. set_ylabel ('MFCC 1 Amplitude', color='blue' )
    
    axes [2]. set (title='MFCC Low-Order Coefficients (Prosody/Intensity Analysis)')
    axes [2].set_xlabel (' Time (s)')

    plt. tight_layout ()
    plt. show ()

    print("¥n --- 分析のポイントーー-")
    print(" -MFCCs(B):縦軸が音色(母音)の特徴を表します。時間方向の変化を見ることで、音色の変化（母音の切り替わり）を追うことができます。")
    print(" -MFCC 0(C,赤線) :** 音量の大小 ** や ** 有声区間 ** に対応しており、 ** 抑揚 ** の基本的な強さを分析するのに役立ちます。")
    print(" - MFCC 1(C,青線) :** 声道形状の基本的な傾き ** を表し、母音の種類(例:[i]と[a]の違い)や、抑揚に伴う音色の変化を捉えます。")

except FileNotFoundError:
    print(f"エラー:ファイルが見つかりません:{file_path}")
except Exception as e:
    print(f"エラーが発生しました:{el")

(A)スペクトログラム(元の周波数情報)