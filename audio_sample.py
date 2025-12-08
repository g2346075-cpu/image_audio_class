import numpy as np
from scipy. io import wavfile
from scipy. signal import stft, windows, get_window, freqz
from scipy. linalg import toeplitz, solve
import matplotlib. pyplot as plt
import IPython
import librosa
import librosa. display

def analyze_audio_with_fft(file_path):

    WAVファイルを読み込み、フーリエ変換(FFT)を実行し、結果をプロットする関数。

    Args:
        file_path (str):分析したいWAVファイルのパス。

     try:

         # 1. WAVファイルを読み込む
         rate:サンプリング周波数(Hz)
         # data: 音声データ(配列)
         rate, data = wavfile. read(file_path)

         # モノラルにする(多チャンネルの場合は最初のチャンネルを使用)
         if data. ndim > 1:
             data = data[:, 0]
         #データの長さ
         N = len (data)
         # 時間軸の作成
         time = np. arange (N) / rate
         # 2. フーリエ変換(FFT)の実行
         #np.fft.fft は複素数の配列を返す
         fft_result = np. fft. fft (data)

         # 3. 周波数軸の作成
         # np.fft.fftfreq はFFT結果に対応する周波数軸を生成する
         freq = np. fft. fftfreq(N, d=1/rate)

         # 4. プロットの準備
         # FFTの結果は左右対称なので、片側(正の周波数)だけを使用することがー
         # n_half: データ点数の半分
         n_half = N // 2

         #振幅スペクトル(絶対値)を計算し、正規化する(任意)
         # dBスケールでプロットする場合は 20*log10(abs(fft_result))などを使
         amplitude_spectrum = np. abs (fft_result[:n_half]) * 2 / N
         # 周波数軸も半分まで
         freq_half = freq[:n_half]

         # 5. 結果のプロット
         plt.figure(figsize=(12, 6))

         # --- 時間領域のプロット(元の波形)
         plt. subplot (2, 1, 1)
         plt.plot (time, data)
         plt. title (' Time Domain: Original Audio Waveform')
         plt. xlabel (' Time (s)')
         plt. ylabel (' Amplitude' )
         plt.xlim(0,rate/2) #プロット範囲をナイキスト周波数までに制限
         plt.grid()

         plt. tight_layout ()
         plt. show ()

         print (f"Sampling Rate: {rate} Hz")
         print (f"Data Points: {N] ")

     except FileNotFoundError:
         print (f"Error: File not found at {file_path]")
     except Exception as e:
         print (f"An error occurred: {e]")

def analyze_audio_with_stft(file_path, start_time=1.0, window_size_ms=30, overlap_rate=0.5):

    WAVファイルを読み込み、短時間フーリエ変換(STFT)を実行し、
    スペクトログラムをプロットする関数。

    Args:
        file_path(str):分析したいWAVファイルのパス。
        window_size_ms (int):フレームの長さ(ミリ秒)。
        overlap_rate(float): フレーム間のオーバーラップ率 (0.0から1.0)。
    try:       
        # WAVファイルを読み込む
        # rate: サンプリング周波数(Hz)
        # data: 音声データ(配列)
        rate, data = wavfile. read(file_path)

        # パラメータの計算
        # フレームの長さ(サンプル数)
        #例えば、サンプリング周波数 44100Hz で 30ms なら、44100 *0.03=1323 サンプル
        window_length = int (rate * window_size_ms / 1000)

        # オーバーラップ(移動)ステップ(サンプル数)
        # 50%オーバーラップの場合、window_length * (1-0.5)
        step_length = int(window_length * (1 - overlap_rate))

        # モノラルにする(多チャンネルの場合は最初のチャンネルを使用)
        if data. ndim > 1:
            data = data[:, 0]

        # データ切り出し
        data = data[int(rate*start_time):int (rate*start_time)+window_length]

        # データの長さ
        N = len (data)
        # 時間軸の作成
        time = np. arange (N) / rate

        # 2. フーリエ変換(FFT)の実行
        # np.fft.fft は複素数の配列を返す
        fft_result = np. fft. fft (data)

        # 3. 周波数軸の作成
        # np.fft.fftfreq はFFT結果に対応する周波数軸を生成する
        freq = np. fft. fftfreq(N, d=1/rate)

        #4. プロットの準備
        # FFTの結果は左右対称なので、片側(正の周波数)だけを使用することが一般的
        # n_half: データ点数の半分
        n_half = N // 2

        # 振幅スペクトル(絶対値)を計算し、正規化する(任意)
        # dBスケールでプロットする場合は 20*log10(abs(fft_result))などを使う
        amplitude_spectrum = np. abs (fft_result[:n_half]) * 2 / N
        #周波数軸も半分まで
        freq_half = freq[:n_half]
        #5. 結果のプロット
        plt.figure(figsize=(12. 12))

        # -== 時間辑域のプロット(元の波形)
        plt.subplot (3. 1. 1)
        plt.plot(time. data)
        plt. title( 'Time Domain: Original Audio Waveform')
        plt. xlabel( 'Time (s)')
        plt. ylabel ( 'Amplitude')
        plt.grid()

        # --- 周波数領域のプロット(振幅スペクトル〕
        plt.subplot (3. 1. 2)
        plt.plot(freg_half. amplitude_spectrum)
        plt.title( 'Frequency Domain: Amplitude Spectrum (FFT)')
        plt.xlabel ( 'Frequency (Hz)')
        plt.ylabel ( 'Amplitude')
        plt.xlim(0. rate/2) #プロット範画をナイキスト開波数までに制限
        plt.grid()

        plt.subolot (3. 1. 3)
        plt.plot(freq_half. amplitude_spectrum)
        plt.title( 'Frequency Domain: Amplitude Spectrum (FFT)')
        plt.xlabel ( 'Frequency (Hz)')
        plt. ylabel ( 'Amplitude')
        plt.xlim(0.500) #ブロット範團をナイキスト周波数までに制限
        plt.grid()
        
        plt.tight_layout()
        plt.show()

        print(f"Samoling Rate: (rate) Hz")
        print(f"Data Points: [N")
        
        data = np.frombuffer(data, dtype="int16")
        
        return IPython. display. Audio(data, rate=rate)

    except FileNotFoundError:
        print(f"Error: File not found at [file_path)")
    except Exception as e:
        print(f"An error occurred: {e}")


    
