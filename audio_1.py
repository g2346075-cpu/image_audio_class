import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def analyze_speech_with_mfccs(file_path, n_mfcc=13):
    """
    MFCCsを抽出し、人の抑揚と母音の特徴を分析するためにプロットする

    Args:
        file_path (str): 分析したいWAVファイルのパス
        n_mfcc (int): 抽出するMFCCsの数
    """
    try:
        # 1. 音声ファイルの読み込み
        x, sr = librosa.load(file_path, sr=None)

        # 2. MFCCsの計算
        mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc,
                                     n_fft=2048, hop_length=512)

        # 時間軸の計算
        times = librosa.frames_to_time(np.arange(mfccs.shape[1]), sr=sr, hop_length=512)

        print("\n--- MFCCs分析情報 ---")
        print(f"MFCCsの形状（係数数 × フレーム数）：{mfccs.shape}")

        # 3. 可視化：スペクトログラムとMFCCsを並列表示
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # --- (A) スペクトログラム（元の周波数情報） ---
        D = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=2048, hop_length=512)),ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz',ax=axes[0], cmap='magma')
        axes[0].set(title='Log Power Spectrogram (For Reference)')
        axes[0].set_ylim(0, sr / 2)
        axes[0].label_outer()

        # --- (B) MFCCs（メル周波数特徴量） ---
        img_mfcc = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[1], cmap='coolwarm')
        axes[1].set(title='MFCCs (Mel Frequency Features for Mother Vowel/Timble Analysis)')
        fig.colorbar(img_mfcc, ax=axes[1], format='%+2.0f dB')
        axes[1].label_outer()

        # --- (C) MFCCsの低次係数（抑揚／音の強さの分析） ---
        # 第0係数：ログエネルギー相当（音の強さ）
        # 第1係数：声道形状の基本的傾き（音色・抑揚）

        # 第0係数（エネルギー）
        axes[2].plot(times, mfccs[0, :], label='MFCC 0 (Log Energy)', color='red', linewidth=2)
        axes[2].set_ylabel('MFCC 0 Amplitude')

        # 第1係数（基本的な音色）
        ax2_twin = axes[2].twinx()
        ax2_twin.plot(times, mfccs[1, :], label='MFCC 1 (Timbre/Slope)', color='blue', linestyle='--', linewidth=1.5)
        ax2_twin.set_ylabel('MFCC 1 Amplitude', color='blue')

        axes[2].set_title('MFCC Low-Order Coefficients (Prosody / Intonation Analysis)')
        axes[2].set_xlabel('Time (s)')

        plt.tight_layout()
        plt.show()

        print("\n--- 分析のポイント ---")
        print(" - MFCCs (B)：縦軸が音色（母音）の特徴を表します")
        print(" - MFCC 0 (C, 赤線)：音量の大小や有声音区間を表します")
        print(" - MFCC 1 (C, 青線)：声道形状の基本的な傾きを表します")

    except Exception as e:
        print("エラーが発生しました:", e)

import librosa
import numpy as np

# -----------------------------
# 発話時間の分析
# -----------------------------
def analyze_duration(file_path):
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    return duration


# -----------------------------
# ピッチ（基本周波数）の分析
# -----------------------------
def analyze_pitch(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # librosaのpyinを使用
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )

    # 無音部分を除外
    f0_clean = f0[~np.isnan(f0)]

    return {
        "mean_f0": np.mean(f0_clean),
        "std_f0": np.std(f0_clean)
    }


# -----------------------------
# MFCC（平均ベクトル）
# -----------------------------
def analyze_mfcc_mean(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean


# -----------------------------
# MFCC距離（話者差）
# -----------------------------
def mfcc_distance(mfcc1, mfcc2):
    return np.linalg.norm(mfcc1 - mfcc2)
