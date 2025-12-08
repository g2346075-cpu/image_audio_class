import numpy as np
import librosa
import librosa.display
import matplotlib. pyplot as plt

def analyze_accent_and_vowels_with_graphs(file_path, n_mfcc=13) :

訛り(アクセント、母音·子音の発音特徴)分析に役立つ複数のグラフを表示する関数。

try:

# 1. 音声ファイルの読み込み
x, sr = librosa. load (file_path, sr=None)

#2. 短時間フーリエ変換(STFT)
# スペクトログラムとピッチ推定に使用
n_fft = 2048
hop_length = 512

D = librosa. stft(x, n_fft=n_fft, hop_length=hop_length)
S_db =librosa.amplitude_to_db(np.abs(D),ref=np.max)#デシベルスケールのパワースペクトログラム

#3. ピッチ(F0)の推定
# librosa.pyinはYINアルゴリズムを実装しており、頑健なピッチ推定が可能
f0, voiced_flag, voiced_probs = librosa. pyin(x, sr=sr, fmin=librosa. note_to_hz ( C2' ), fmax=librosa.note_to_hz('C5))
f0_times = librosa. times_like(f0, sr=sr, hop_length=hop_length)

# 4. MFCCs(メル周波数ケプストラム係数)の計算
mfccs = librosa. feature. mfcc (y=x, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
mfcc_times = librosa. frames_to_time (np. arange (mfccs. shape[1]), sr=sr, hop_length=hop_length)

# 5. RMSエネルギー(音の強さ)の計算
rms = librosa. feature. rms (y=x, frame_length=n_fft, hop_length=hop_length) [0]
rms_times = librosa. times_like(rms, sr=sr, hop_length=hop_length)

# 6. 可視化:複数のサブプロットで情報を表示
fig, axes = plt. subplots(4, 1, figsize=(12, 12), sharex=True)# 4つのグラフを縦に並べる

(A)音声波形とRMSエネルギー
# 音声波形
librosa. display. waveshow(x, sr=sr, ax=axes [0], color=' lightgray' )
# RMSエネルギーを重ねて表示
axes [0].plot(rms_times, rms, color='blue', label='RMS Energy (dB)', alpha=0.8)
axes [0]. set (title='Audio Waveform & RMS Energy', ylabel=' Amplitude / RMS' )
axes [0]. grid(True, linestyle=' :' )
axes [0]. legend ( loc=' upper right' )

# --- (B)対数パワー·スペクトログラム-
img_spec = librosa. display. specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=axes [1], cmap='magma')
axes [1]. set(title='Log Power Spectrogram (Overall Spectral Content)', ylabel='Frequency (Hz)')
axes[1].set_ylim(0, sr/2)#最大周波数をサンプリングレートの半分に設定
fig. colorbar (img_spec, ax=axes[1], format="%+2. Of dB")

-(C)ピッチトラッキング(F0)
# NaNを除外してプロット(無声区間はプロットしない)
axes[2]. plot(f0_times, f0, color='cyan', linewidth=2, label=' Estimated Pitch (F0)')
axes[2]. set (title='Pitch Tracking (FO) (Intonation & Accent Analysis)', ylabel=' Frequency (Hz)' )
axes[2].set_ylim(50, 400)#人間の声のF0範囲
axes[2]. grid(True, linestyle=' :' )
axes[2]. legend (loc=' upper right' )

# --- (D)MFCCs(メル周波数ケプストラム係数)
img_mfcc = librosa. display. specshow(mfccs, sr=sr, x_axis='time', y_axis='mel', ax=axes [3], cmap='coolwarm')
axes [3]. set(title=f' MFCCs (Timbre, Vowel/Consonant Articulation)')
axes [3].set_xlabel (' Time (s)')
fig. colorbar (img_mfcc, ax=axes[3], format="%+2. Of")

plt. tight_layout ()
plt. show ()

# -