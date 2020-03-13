import numpy as np
import sys
import matplotlib.pyplot as plt
import pdb
import matplotlib.colors as colors
from scipy import fftpack


# Get Image Name
def getImageString():
    for i in range(len(sys.argv)):
        if sys.argv[i] == "-i":
            try:
                f = open(sys.argv[i+1])
                image = sys.argv[i+1]
                f.close
            except IOError:
                print("Image not accessible")
                exit(1)
    return plt.imread(image).astype(float)

# Get Mode Number


def getModeNumber():
    for i in range(len(sys.argv)):
        if sys.argv[i] == "-m":
            if (sys.argv[i+1] == "1" or sys.argv[i+1] == "2" or sys.argv[i+1] == "3" or sys.argv[i+1] == "4"):
                mode = sys.argv[i+1]
            else:
                print("Mode number should be either 1 or 2 or 3 or 4")
                exit(1)
    return int(mode)


# 1D array

def naiveDFT_1D(x):
    x = np.asarray(x, dtype=float)
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    exp = np.exp(-2j * np.pi * k * n / N)
    ft = np.dot(exp, x)
    return ft


def fft_1D(x):
    n = len(x)
    x = x.astype(dtype=float)
    k = np.arange(n)
    if (n % 2 != 0):
        print("X should be power of 2")
        exit(1)
    elif (n <= 4):
        return naiveDFT_1D(x)
    else:
        X_even = fft_1D(x[0::2])
        X_odd = fft_1D(x[1::2])
        factor = np.exp(-2j * np.pi * k / n)
        X_even = np.concatenate([X_even, X_even])
        X_odd = np.concatenate([X_odd, X_odd])
        return X_even + factor * X_odd


def naiveIDFT_1D(x):
    x = np.asarray(x, dtype=float)
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    exp = np.exp(2j * np.pi * k * n / N)
    ft = np.dot(exp, x)
    return ft


def prepare(x):
    n = x.shape[0]
    x = np.asarray(x, dtype=float)
    k = np.arange(n)
    if (n % 2 != 0):
        print("X should be power of 2")
        exit(1)
    elif (n <= 32):
        return naiveIDFT_1D(x)
    else:
        X_even = n//2 * ifft_1D(x[0::2])
        X_odd = n//2 * ifft_1D(x[1::2])
        factor = np.exp(2j * np.pi * k / n)
        X_even = np.concatenate([X_even, X_even])
        X_odd = np.concatenate([X_odd, X_odd])
        return X_even + factor * X_odd


def ifft_1D(x):
    return (1/len(x))*prepare(x)


# 2D array

def fft_2D(a):
    x = np.zeros(a.shape, dtype=complex)
    M, N = a.shape
    for i in range(N):
        x[:, i] = fft_1D(a[:, i])
    for i in range(M):
        x[i, :] = fft_1D(x[i, :])
    return x


def ifft_2D(a):
    x = np.zeros(a.shape, dtype=complex)
    M, N = a.shape
    for i in range(M):
        x[i, :] = ifft_1D(a[i, :])
    for i in range(N):
        x[:, i] = ifft_1D(x[:, i])
    return x
