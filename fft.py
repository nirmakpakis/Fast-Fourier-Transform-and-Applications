import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors


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
    return image

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


# Pad Image

def padImage(image):
    result = np.zeros((next_power_of_2(
        image.shape[0]), next_power_of_2(image.shape[1])))
    result[:image.shape[0], :image.shape[1]] = image
    return result


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


# 1D array
def naiveDFT_1D(x):
    x = np.asarray(x, dtype=complex)
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    exp = np.exp(-2j * np.pi * k * n / N)
    ft = np.dot(exp, x)
    return ft


def fft_1D(x):
    n = len(x)
    x = x.astype(dtype=complex)
    k = np.arange(n)
    if (n <= 32):
        return naiveDFT_1D(x)
    else:
        X_even = fft_1D(x[0:: 2])
        X_odd = fft_1D(x[1:: 2])
        factor = np.exp(-2j * np.pi * k / n)
        X_even = np.concatenate([X_even, X_even])
        X_odd = np.concatenate([X_odd, X_odd])
        return X_even + factor * X_odd


def naiveIDFT_1D(x):
    x = np.asarray(x, dtype=complex)
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    exp = np.exp(2j * np.pi * k * n / N)
    ft = np.dot(exp, x)
    return ft


def prepare(x):
    n = x.shape[0]
    x = np.asarray(x, dtype=complex)
    k = np.arange(n)
    if (n <= 32):
        return naiveIDFT_1D(x)
    else:
        X_even = n//2 * ifft_1D(x[0:: 2])
        X_odd = n//2 * ifft_1D(x[1:: 2])
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


# MODES

def modeOne():
    # create a template for 2 images
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    # add original image
    originalImage = plt.imread(getImageString()).astype(float)
    plt.title('Original Image')
    plt.imshow(originalImage, cmap="gray")
    # add Fourier Transform of the image
    fftImage = fft_2D(padImage(originalImage))
    f.add_subplot(1, 2, 2)
    plt.title('Fourier transform')
    plt.imshow(np.abs(fftImage), norm=colors.LogNorm(vmin=5), cmap='gray')
    plt.colorbar()
    # print image
    plt.show(block=True)


def modeTwo():
    # create a template for 2 images
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    # add original image
    originalImage = plt.imread(getImageString()).astype(float)
    plt.title('Original Image')
    plt.imshow(originalImage, cmap="gray")
    # add denoised image
    denoisedImage = denoiseImage(originalImage)[0:len(
        originalImage), 0:len(originalImage[0])]
    f.add_subplot(1, 2, 2)
    plt.title('Denoised Image')
    plt.imshow(denoisedImage, cmap='gray')
    # print image
    plt.show(block=True)


def denoiseImage(image):
    fftImage = fft_2D(padImage(image))
    keep_fraction = 0.05
    r, c = fftImage.shape
    fftImage[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
    fftImage[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
    filteredImage = np.abs(ifft_2D(fftImage))
    return filteredImage


def modeThree():
    # create a template for 2 images
    f = plt.figure()
    f.add_subplot(2, 3, 1)
    # add original image
    originalImage = plt.imread(getImageString()).astype(float)
    plt.title('Original Image')
    plt.imshow(originalImage, cmap="gray")
    # add image
    f.add_subplot(2, 3, 2)
    plt.title('%15 Trunctated')
    image = compress(originalImage, 15)
    plt.imshow(image, cmap='gray')
    # add image
    f.add_subplot(2, 3, 3)
    plt.title('%30 Trunctated')
    image = compress(originalImage, 30)
    plt.imshow(image, cmap='gray')
    # add image
    f.add_subplot(2, 3, 4)
    plt.title('%50 Trunctated')
    image = compress(originalImage, 50)
    plt.imshow(image, cmap='gray')
    # add image
    f.add_subplot(2, 3, 5)
    plt.title('%75 Trunctated')
    image = compress(originalImage, 75)
    plt.imshow(image, cmap='gray')
    # add image
    f.add_subplot(2, 3, 6)
    plt.title('%95 Trunctated')
    image = compress(originalImage, 95)
    plt.imshow(image, cmap='gray')
    # print image
    plt.show(block=True)


def compress(image, percentage):
    p = np.percentile(image, percentage)
    r, c = image.shape
    for i in range(r):
        for j in range(c):
            if image[i][j] < p:
                image[i][j] = 0
    return image


modeThree()
