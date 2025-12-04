import wave
import numpy as np
import matplotlib.pyplot as plt

wav_file = wave.open("./audio/input.wav", "r")
buf = wav_file.readframes(wav_file.getnframes())
data = np.frombuffer(buf, dtype="int16")

print("全区域")
plt.plot(data)
plt.grid()
plt.show()

print("特定区間を切り出す")
plt.plot(data)
plt.xlim([50000, 51000])
plt.show()


def show():
    wav_file = wave.open("./audio/input.wav", "r")
    buf = wav_file.readframes(wav_file.getnframes())
    data = np.frombuffer(buf, dtype="int16")

    print("全区域")
    plt.plot(data)
    plt.grid()
    plt.show()
    
    print("特定期間を切り出す")
    plt.plot(data)
    plt.xlim([50000,51000])
    plt.show()
