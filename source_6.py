import numpy as np
import IPython
import matplotlib.pyplot as plt

# 440を基準に、音ずらす
freqs = [0] + [440.0 * 2.0**((i - 9) / 12.0) for i in range(12)]

# 角周波数に対応コードを生成する。
notes = ["R", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
dic = {}
for i, s in enumerate(notes):
    dic[s] = i

print(freqs)
print(dic)


def mml(mml):
    rate = 48000
    BPM = 120
    qn_duration = 60.0 / BPM
    t = np.linspace(0.0, qn_duration, int(rate * qn_duration))
    music = np.array([])

    for s in list(mml):
        f = freqs[dic[s]]
        music = np.append(music, np.sin(2.0 * np.pi * f * t))

    print("全区域")
    plt.plot(music)
    plt.grid()
    plt.show()

    return IPython.display.Audio(music, rate=rate, autoplay=True)
