import numpy as np
import IPython
import matplotlib.pyplot as plt

def play():
    rate = 48000
    duration = 2.0
    t = np.linspace(0., duration, int(rate * duration))
    x1 = np.sin(2.0 * np.pi * 440.0 * t)
    x2 = np.sin(2.0 * np.pi * 880.0 * t)

    x = x1 + x2

    print("全区域")
    plt.plot(x)
    plt.grid()
    plt.show()

    print("特定区間を切り出す")
    plt.plot(x, 'o')
    plt.xlim([0, 440])
    plt.show()

    print("特定区間を切り出す")
    plt.plot(x, 'o')
    plt.xlim([25, 50])
    plt.show()

    return IPython.display.Audio(x, rate=rate, autoplay=True)
