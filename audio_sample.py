import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, windows, get_window, freqz
from scipy.linalg import toeplitz, solve
import matplotlib.pyplot as plt
import IPython
import librosa
import librosa.display


def analyze_audio_with_fft(file_path):
    """
    WAVファイルを読み込み、フーリエ変換（FFT）を実行し、結果をプロットする関数。

    Args:
        file_path (str): 分析したいWAVファイルのパス。
    """
    try:
        # 1. WAVファイルを読み込む
        rate, data = wavfile.read(file_path)  # rate: サンプリング周波数, data: 音声データ（配列）

        # モノラルにする（多チャンネルの場合は最初のチャンネルを使用）
        if data.ndim > 1:
            data = data[:, 0]

        # 2. データの長さ
        N = len(data)
        # 時間軸の作成
        time = np.arange(N) / rate

        # 3. フーリエ変換（FFT）の実行
        fft_result = np.fft.fft(data)

        # 4. 周波数軸の作成
        freq = np.fft.fftfreq(N, d=1 / rate)

        # 5. プロット用に片側だけ使う
        n_half = N // 2
        freq_half = freq[:n_half]
        amplitude_spectrum = np.abs(fft_result[:n_half]) * 2 / N

        # 6. 結果のプロット
        plt.figure(figsize=(12, 6))

        # --- 時間領域のプロット（元の波形） ---
        plt.subplot(2, 1, 1)
        plt.plot(time, data)
        plt.title('Time Domain: Original Audio Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid()

        # --- 周波数領域のプロット（振幅スペクトル） ---
        plt.subplot(2, 1, 2)
        plt.plot(freq_half, amplitude_spectrum)
        plt.title('Frequency Domain: Amplitude Spectrum (FFT)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(0, rate / 2)  # ナイキスト周波数まで
        plt.grid()

        plt.tight_layout()
        plt.show()

        print(f"Sampling Rate: {rate} Hz")
        print(f"Data Points: {N}")

        # ついでに音も返したい場合（お好みで）
        return IPython.display.Audio(data, rate=rate)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def analyze_audio_with_stft(file_path, start_time=1.0, window_size_ms=30, overlap_rate=0.5):
    """
    WAVファイルを読み込み、短時間フーリエ変換（STFT）を実行し、
    ウィンドウを切り出して FFT をプロットする簡易版関数。

    Args:
        file_path (str): 分析したいWAVファイルのパス。
        window_size_ms (int): フレームの大きさ（ミリ秒）
        overlap_rate (float): フレーム間のオーバーラップ率（0.0〜1.0）
    """
    try:
        # 1. WAVファイルを読み込む
        rate, data = wavfile.read(file_path)

        # フレームの大きさ（サンプル数）
        window_length = int(rate * window_size_ms / 1000)

        # オーバーラップ量（使うならここで step_length を計算）
        step_length = int(window_length * (1 - overlap_rate))

        # モノラル化
        if data.ndim > 1:
            data = data[:, 0]

        # 指定時刻から window_length 分だけ切り出す
        start_idx = int(rate * start_time)
        data = data[start_idx:start_idx + window_length]

        # データ長と時間軸
        N = len(data)
        time = np.arange(N) / rate

        # FFT
        fft_result = np.fft.fft(data)
        freq = np.fft.fftfreq(N, d=1 / rate)
        n_half = N // 2
        freq_half = freq[:n_half]
        amplitude_spectrum = np.abs(fft_result[:n_half]) * 2 / N

        # プロット
        plt.figure(figsize=(12, 12))

        # --- 時間領域 ---
        plt.subplot(3, 1, 1)
        plt.plot(time, data)
        plt.title('Time Domain: Windowed Audio')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid()

        # --- 全周波数 ---
        plt.subplot(3, 1, 2)
        plt.plot(freq_half, amplitude_spectrum)
        plt.title('Frequency Domain: FFT Amplitude Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(0, rate / 2)
        plt.grid()

        # --- 低周波（0〜500Hz） ---
        plt.subplot(3, 1, 3)
        plt.plot(freq_half, amplitude_spectrum)
        plt.title('Frequency Domain: FFT (Low Frequency)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(0, 500)
        plt.grid()

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def analyze_audio_with_stft_window(file_path, start_time=1.0, window_size_ms=30, overlap_rate=0.5):

    """
    WAVファイルを読み込み、短時間フーリエ変換（STFT）を実行し、
    スペクトログラムをプロットする関数。
    
    Args:
        file_path (str): 分析したいWAVファイルのパス。
        window_size_ms (int): フレームの大きさ（ミリ秒）
        overlap_rate (float): フレーム間のオーバーラップ率（0.0から1.0）。
    """
    try:
        # 1. WAVファイルを読み込む
        # rate: サンプリング周波数 (Hz)
        # data: 音声データ（配列）
        rate, data = wavfile.read(file_path)

        # パラメータの計算
        # フレームの大きさ（サンプル数）
        # 例：44100Hz、サンプリング周波数 44100で 30ms なら、44100 * 0.03 = 1323 サンプル
        window_length = int(rate * window_size_ms / 1000)

        # オーバーラップ（移動）ステップ（サンプル数）
        # 50%オーバーラップの場合、window_length * (1 - 0.5)
        step_length = int(window_length * (1 - overlap_rate))

        # モノラルにする（多チャンネルの場合は最初のチャンネルを使用）
        if data.ndim > 1:
            data = data[:, 0]

        # データ切り出し
        data = data[int(rate*start_time):int(rate*start_time)+window_length]

        # データの長さ
        N = len(data)
        # 時間軸の作成
        time = np.arange(N) / rate

        # 3. 窓関数の作成と適用
        # 一般的な窓関数（Hann Window）を使用
        hann_window = windows.hann(N, sym=True)
        windowed_data = data * hann_window

        # 2. フーリエ変換（FFT）の実行
        fft_result = np.fft.fft(windowed_data)

        # 3. 周波数軸の作成
        freq = np.fft.fftfreq(N, d=1/rate)

        # 4. プロットの準備
        # FFT結果は左右対称なので、片側（正の周波数）だけを使用することが一般的
        n_half = N // 2
        freq_half = freq[:n_half]

        # 振幅スペクトル（絶対値）を計算し、正規化する（任意）
        amplitude_spectrum = np.abs(fft_result[:n_half]) * 2 / N

        # 5. 結果のプロット
        plt.figure(figsize=(12, 12))

        # --- 時間領域のプロット（元の波形） ---
        plt.subplot(4, 1, 1)
        plt.plot(time, data)
        plt.title('Original Frame (N samples)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid()

        # --- 時間領域のプロット（窓関数） ---
        plt.subplot(4, 1, 2)
        plt.plot(time, windowed_data)
        plt.title('Windowed Frame (N samples, Hann Window)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid()

        # --- 窓関数のみのプロット（参考） ---
        plt.subplot(4, 1, 3)
        plt.plot(hann_window)
        plt.title('Hann Window Function')
        plt.xlabel('Sample Index')
        plt.grid()

        # --- 周波数領域のプロット（振幅スペクトル） ---
        plt.subplot(4, 1, 4)
        plt.plot(freq_half, amplitude_spectrum)
        plt.title('Frequency Domain: Amplitude Spectrum (FFT)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(0, 500)  # プロット範囲をナイキスト周波数までに制限
        plt.grid()

        plt.tight_layout()
        plt.show()

        print(f"Sampling Rate: {rate} Hz")
        print(f"Data Points: {N}")

        data = np.frombuffer(data, dtype="int16")
        return IPython.display.Audio(data, rate=rate)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
def calculate_cepstrum(frame, sr):
    """ 単一フレームに対してケプストラムを計算する。 """

    N = len(frame)
    eps = 1e-10

    # 1. FFT -> log(|FFT(x)|)
    fft_result = np.fft.fft(frame, N)
    abs_result = np.abs(fft_result + eps)
    log_magnitude = 20 * np.log10(abs_result / np.max(abs_result))

    # 2. IFFT -> ケプストラム
    cepstrum = np.real(np.fft.ifft(log_magnitude))
    cepstrum[100:] = 0   # ローパスリフタ（cepstrumの高次成分をゼロ）

    plt.figure(figsize=(10, 6))
    plt.plot(cepstrum, label='Cepstrum', color='gray', alpha=0.5)
    plt.grid()

    # 3. 対数パワースペクトルの再構築
    fft_cepstrum = np.real(np.fft.fft(cepstrum, N))

    return log_magnitude, fft_cepstrum

def calculate_noise(frame, sr):
    """ 単一フレームに対してノイズケプストラムを計算する。 """

    N = len(frame)
    eps = 1e-10

    # 1. FFT -> log(|FFT(x)|)
    fft_result = np.fft.fft(frame, N)
    abs_result = np.abs(fft_result + eps)
    log_magnitude = 20 * np.log10(abs_result / np.max(abs_result))

    # 2. IFFT -> ケプストラム
    cepstrum = np.real(np.fft.ifft(log_magnitude))
    cepstrum[25:100] = 0  # ローパスリフタ

    plt.figure(figsize=(10, 6))
    plt.plot(cepstrum, label='Noise Spectrum', color='gray', alpha=0.5)
    plt.grid()

    fft_cepstrum = np.real(np.fft.fft(cepstrum, N))

    return log_magnitude, fft_cepstrum

def analyze_and_plot_signal_and_cepstrum(file_path, start_time_sec=0.5, frame_len_ms=30, n_fft=2048):

    try:
        # 1. WAVファイルの読み込みとデータの取得
        rate, data = wavfile.read(file_path)
        sr = rate
        data = data.astype(np.float64)
        if data.ndim > 1:
            data = data[:, 0]

        # 2. フレーム長の計算と切り出し位置
        N_frame = int(sr * frame_len_ms / 1000)
        start_sample = int(start_time_sec * sr)

        # FFTサイズに合わせてフレームをパディング
        N = max(N_frame, n_fft)

        frame = data[start_sample : start_sample + N_frame]
        if len(frame) < N_frame:
            raise ValueError("指定されたフレーム長が音声データを超えています。")

        # === FFTスペクトル ===
        freq_fft = np.fft.fftfreq(N, 1 / sr)
        freq_fft = freq_fft[:N//2]

        # FFTサイズに合わせてフレームをパディング（ゼロ埋め）
        padded_frame = np.pad(frame, (0, N - len(frame)), 'constant')

        window = get_window('hann', N, fftbins=True)
        windowed_frame = padded_frame * window
        # 3. ケプストラムの計算
        log_magnitude, cepst_power = calculate_cepstrum(windowed_frame, sr)

        plt.figure(figsize=(10, 6))
        plt.plot(freq_fft, log_magnitude[:N//2], label='FFT Spectrum', color='gray', alpha=0.5)
        plt.plot(freq_fft, cepst_power[:N//2], label='Cepstrum', color='red', alpha=0.5)
        plt.xlabel('Power (dB)')
        plt.xlim(0, rate/2)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 4. ケプストラム以外の成分の計算
        log_magnitude, ceps_power = calculate_noise(windowed_frame, sr)

        plt.figure(figsize=(10, 6))
        plt.plot(freq_fft, log_magnitude[:N//2], label='FFT Spectrum', color='gray', alpha=0.5)
        plt.plot(freq_fft, ceps_power[:N//2], label='Noise', color='red', alpha=0.5)
        plt.xlabel('Power (dB)')
        plt.xlim(0, rate/2)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"エラー：ファイルが見つかりません: {file_path}")
    except ValueError as e:
        print(f"値のエラー：{e}")
    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}")

def extract_formants(file_path, start_time_sec=0.5, frame_len_ms=25, lpc_order=20):
    """
    numpyとscipyのみを使用して、LPCによるフォルマント周波数を推定・プロットする関数。

    Args:
        file_path (str): 分析したいWAVファイルのパス。
        start_time_sec (float): 分析を開始する時間（秒）
        frame_len_ms (int): 解析するフレームの長さ（ミリ秒）
        lpc_order (int): LPC分析の次数
    """
    try:
        # 1. WAVファイルの読み込みとデータの取得
        rate, data = wavfile.read(file_path)
        sr = rate
        data = data.astype(np.float64)
        if data.ndim > 1:
            data = data[:, 0]

        # 2. パラメータの計算とフレームの抽出
        N = int(sr * frame_len_ms / 1000)   # フレーム長（サンプル数）
        start_sample = int(start_time_sec * sr)
        end_sample = start_sample + N

        if end_sample > len(data):
            print(f"エラー：データが短すぎます。音声の長さ：{len(data)} サンプル")
            return

        frame = data[start_sample:end_sample]

        # 3. 窓関数の適用（ハミング窓を使用）
        window = get_window('hamming', N, fftbins=True)
        windowed_frame = frame * window

        # 3. LPC係数の計算（オートコレレーション）
        R = np.correlate(windowed_frame, windowed_frame, mode='full')
        R = R[len(frame)-1 : len(frame) + lpc_order]   # R[0] 〜 R[p] まで

        # トープリッツ行列とスペクトルの作成
        A = toeplitz(R[:lpc_order])
        r = R[1 : lpc_order + 1]

        # 線形方程式 Ax = b を解く → LPC係数
        lpc_coeff = np.hstack(([1.0], solve(A, r)))

        # 4. フォルマント（共振周波数）の推定
        roots = np.roots(lpc_coeff)   # LPC係数の根
        roots = roots[np.imag(roots) >= 0]  # 上半平面の根のみを抽出

        # 誤差のある根を除去し周波数を計算
        angles = np.angle(roots)
        formants = angles * (sr / (2.0 * np.pi))

        # フォルマント周波数の昇順にソート
        formants = np.sort(formants)

        # F1 と F2 の決定
        F1 = formants[0] if len(formants) > 0 else np.nan
        F2 = formants[1] if len(formants) > 1 else np.nan

        # 5. 結果のプロット（matplotlibのみを使用）

        plt.figure(figsize=(10, 6))

        # --- A. フレームのFFTスペクトル ---
        fft_frame = np.abs(np.fft.fft(windowed_frame, N))
        freq_fft = np.fft.fftfreq(N, 1/sr)
        fft_db = 20 * np.log10(fft_frame[:N//2] / np.max(fft_frame[:N//2]))
        freq_fft = freq_fft[:N//2]

        plt.plot(freq_fft, fft_db, label='FFT Spectrum', color='gray', alpha=0.5)

        # --- B. LPCによる音声フィルタの周波数応答 ---
        w, h = freqz([1], lpc_coeff, worN=N//2, fs=sr)
        h_db = 20 * np.log10(np.abs(h) / np.max(np.abs(h)))

        plt.plot(w, h_db, label='LPC Envelope (Vocal Tract)', color='blue', linewidth=2)

        # --- C. F1・F2 のプロット ---
        plt.axvline(x=F1, color='red', linestyle='--', label=f'F1_is = {F1:.1f} Hz')
        plt.axvline(x=F2, color='green', linestyle='--', label=f'F2_is = {F2:.1f} Hz')

        plt.title(f'Formant Extraction (LPC) at {start_time_sec} sec')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.xlim(0, sr/2)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("\n--- Estimated Formants ---")
        print(f"F1: {F1:.1f} Hz")
        print(f"F2: {F2:.1f} Hz")

        return IPython.display.Audio(frame, rate=rate)

    except FileNotFoundError:
        print(f"エラー：ファイルが見つかりません: {file_path}")
    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}")

def calculate_and_plot_spectrogram(file_path, window_size_ms=300, overlap_rate=0.5):
    """
    numpyとscipyのみを使用してSTFTを実装し、スペクトログラムをプロットする関数。

    Args:
        file_path (str): 分析したいWAVファイルのパス。
        window_size_ms (int): フレームの大きさ（オーバーラップ率に影響）。
    """
    try:
        # 1. WAVファイルの読み込みとデータの処理
        rate, data = wavfile.read(file_path)
        sr = rate
        data = data.astype(np.float64)
        if data.ndim > 1:
            data = data[:, 0]

        # 2. パラメータの計算とフレームの抽出
        N = int(sr * window_size_ms / 1000)    # フレーム長（サンプル数）
        hop_length = int(N * (1 - overlap_rate))

        # 2. パラメータの定義
        window = get_window('hann', N, fftbins=True)   # ハニング窓関数

        # 3. STFTの実行（手動でのフレーム処理+FFT）

        # 信号の全サンプル数
        total_samples = len(data)
        # STFTフレームの開始インデックス
        start_indices = np.arange(0, total_samples - N, hop_length)
        # フレーム数
        num_frames = len(start_indices)

        # 周波数ビン → フレーム × の形に変換
        # FFTは左右対称構造なので、片側（正の周波数）だけを使用
        fft_bins = N // 2 + 1
        D = np.zeros((fft_bins, num_frames), dtype=complex)

        # フレームごとに処理
        for i, start in enumerate(start_indices):
            # フレーム抽出
            frame = data[start:start + N]

            # 窓関数の適用
            windowed_frame = frame * window

            # FFTの実行
            fft_result = np.fft.fft(windowed_frame, N)

            # 振幅スペクトル（正の周波数）を格納
            D[:, i] = fft_result[:fft_bins]

        # 4. スペクトログラムの計算
        # 複素スペクトルをデシベルスケールに変換
        magnitude = np.abs(D)
        # dBに変換（0dBに正規化しない。ここでは最大値を基準に正規化。）
        D_db = 20 * np.log10(magnitude / np.max(magnitude))

        # 5. 時間と周波数軸の作成
        # 時間軸（秒）
        times = start_indices / sr
        # 周波数軸
        frequencies = np.linspace(0, sr / 2, fft_bins)

        # 6. プロット（matplotlib のみを使用）
        plt.figure(figsize=(10, 6))

        # pcolormeshでスペクトログラムを描画
        plt.pcolormesh(times, frequencies, D_db, shading='gouraud', cmap='viridis')

        plt.colorbar(label='Power (dB)')
        plt.title('Spectrogram (Manual STFT)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.ylim(0, sr / 2)   # ナイキスト周波数まで制限
        plt.tight_layout()
        plt.show()

        print(f"Sampling Rate (sr): {sr} Hz")
        print(f"FFT Size: {N//2}")
        print(f"Hop Length: {hop_length}")

    except FileNotFoundError:
        print(f"エラー：ファイルが見つかりません: {file_path}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

def plot_spectrogram_librosa(file_path):
    """
    Librosaを使用してWAVファイルを読み込み、スペクトログラムをプロットする関数。

    Args:
        file_path (str): 分析したいWAVファイルのパス。
    """
    try:
        # 1. 音声ファイルの読み込み
        # x: 音声信号（NumPy配列）
        # sr: サンプリング周波数（Hz）
        x, sr = librosa.load(file_path, sr=None)

        # 2. 短時間フーリエ変換（STFT）の実行
        # Dは複素数の行列（周波数ビン × 時間フレーム）
        D = librosa.stft(x)

        # 3. 複素スペクトルをデシベルスケールに変換
        # スペクトログラムでは、人間の聴覚に近い対数スケール（dB）を使うのが一般的
        # ref=np.max は、スペクトルの最大値を0dBとして正規化することを意味します
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # 4. プロット（スペクトログラムの描画）
        plt.figure(figsize=(10, 6))

        # librosa.display.specshowで描画
        # x_axis='time' → 横軸を時間にする
        # y_axis='hz' → 縦軸を周波数（Hz）にする
        librosa.display.specshow(
            D_db,
            sr=sr,
            x_axis='time',
            y_axis='log'   # 縦軸を対数スケールにすると、低周波数の変化が見やすくなります
        )

        plt.colorbar(format='%+2.0f dB', label='Power (dB)')
        plt.title('Spectrogram (STFT)')
        plt.tight_layout()
        plt.show()

        print(f"Sampling Rate (sr): {sr} Hz")
        print(f"Spectrogram shape: {D_db.shape} (Frequency Bins × Time Frames)")

    except FileNotFoundError:
        print(f"エラー：ファイルが見つかりません: {file_path}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

def plot_pitch(file_path, fmin=80, fmax=400):
    """
    音声ファイルからピッチ周期成分を検出し、スペクトログラム上に重ねてプロットする関数。

    Args:
        file_path (str): 分析したいWAVファイルのパス。
        fmin (int): ピッチ推定の最小周波数（低い声の男性：約80Hz〜）。
        fmax (int): ピッチ推定の最大周波数（高い声の女性：最大400Hz）。
    """
    try:
        # --- 1. パラメータ設定 ---
        HOP_LENGTH = 1024

        # 1. 音声ファイルの読み込み
        x, sr = librosa.load(file_path, sr=None)

        # 2. 基本周波数(F0) の推定（YINアルゴリズムを使用）
        # voiced_flag: 発声しているかどうか（True/False の配列）
        # voiced_probs: 発声確率（0〜1）
        f0, voiced_flag, voiced_probs = librosa.pyin(
            x,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=HOP_LENGTH
        )

        # 3. 時間軸の作成
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP_LENGTH)

        # 4. グラフの作成
        plt.figure(figsize=(10, 4))

        # ピッチ推定結果のプロット
        f0_plot = np.where(voiced_flag, f0, np.nan)
        plt.plot(times, f0_plot, label='Estimated Pitch (F0)', color='blue', linewidth=1.5)

        plt.xlabel('Time (s)')
        plt.ylabel('Pitch Frequency (Hz)')
        plt.title('Pitch Frequency (F0) over Time')
        plt.legend()
        plt.grid(True, linestyle='--')

        plt.ylim(fmin, fmax)
        plt.xlim(0, times[-1] if len(times) > 0 else 0)
        plt.tight_layout()
        plt.show()

        print(f"Sampling Rate (sr): {sr} Hz")
        print(f"Pitch Estimation Range: {fmin} 〜 {fmax} Hz")

    except FileNotFoundError:
        print(f"エラー：ファイルが見つかりません: {file_path}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

def create_mel_filterbank(sr, n_fft, n_mels=128, fmin=0, fmax=None):
    """
    メルスケールに基づいて三角フィルタバンクを生成する。
    """
    if fmax is None:
        fmax = sr / 2

    # Hz → Mel 変換
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    # Mel → Hz 変換
    def mel_to_hz(mel):
        return 700.0 * (10.0**(mel / 2595.0) - 1.0)

    # メルスケール上の等間隔な点を定義（n_mels + 2）
    m_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(m_pts)

    # FFTビンに対応する周波数
    bin_frequencies = np.linspace(0, sr/2, n_fft//2 + 1)

    # フィルタバンク行列を作成（n_mels × n_fft//2 + 1）
    mel_filterbank = np.zeros((n_mels, n_fft//2 + 1))

    for i in range(n_mels):
        # 三角窓の中心位置と左右の境界を取得
        left = hz_pts[i]
        center = hz_pts[i + 1]
        right = hz_pts[i + 2]

        # フィルタの値を計算
        for j, freq in enumerate(bin_frequencies):
            if left < freq < center:
                mel_filterbank[i, j] = (freq - left) / (center - left)
            elif center < freq < right:
                mel_filterbank[i, j] = (right - freq) / (right - center)

    return mel_filterbank

def calculate_and_plot_mel_spectrogram_manual(file_path, n_fft=2048, hop_length=512, n_mels=128):
    """
    手動で実装したメルスペクトログラムを計算しプロットする。
    """
    try:
        # 1. WAVファイルの読み込み
        rate, data = wavfile.read(file_path)
        sr = rate
        data = data.astype(np.float64)
        if data.ndim > 1:
            data = data[:, 0]

        # 2. STFTの準備
        window = get_window('hann', n_fft, fftbins=True)
        total_samples = len(data)
        start_indices = np.arange(0, total_samples - n_fft, hop_length)
        num_frames = len(start_indices)

        # パワースペクトル行列を初期化
        fft_bins = n_fft // 2 + 1
        P_spectrum = np.zeros((fft_bins, num_frames))

        # 3. メルフィルタバンクの生成
        mel_filterbank = create_mel_filterbank(sr, n_fft, n_mels=n_mels)

        # 4. フレームごとにSTFTの実行
        for i, start in enumerate(start_indices):
            frame = data[start:start + n_fft]
            windowed_frame = frame * window

            fft_result = np.fft.fft(windowed_frame, n_fft)
            magnitude_spectrum = np.abs(fft_result[:fft_bins])
            P_spectrum[:, i] = magnitude_spectrum**2

        # 5. メルスペクトログラムの計算（メルバンク × パワースペクトル）
        mel_spectrogram_raw = np.dot(mel_filterbank, P_spectrum)
        eps = 1e-10
        mel_spectrogram_db = 10.0 * np.log10(mel_spectrogram_raw + eps)

        # 7. プロット
        times = start_indices / sr

        plt.figure(figsize=(10, 6))
        # pcolormeshでメルスペクトログラムを描画
        plt.pcolormesh(times, np.arange(n_mels), mel_spectrogram_db, cmap='viridis', shading='gouraud')

        plt.colorbar(label='Power (dB)')
        plt.title('Mel Spectrogram (Manual Implementation)')
        plt.xlabel('Time (s)')
        plt.ylabel('Mel Filter Index')
        plt.ylim(0, n_mels)
        plt.tight_layout()
        plt.show()

        print("\n--- メルスペクトログラム情報 ---")
        print(f"サンプリングレート (sr): {sr} Hz")
        print(f"メルスペクトログラムの形状（メルビン × 時間フレーム）: {mel_spectrogram_db.shape}")

    except FileNotFoundError:
        print(f"エラー：ファイルが見つかりません: {file_path}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")