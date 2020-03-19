import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as image
from scipy import sparse
import time

# Helper functions


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


def getModeNumber():
    for i in range(len(sys.argv)):
        if sys.argv[i] == "-m":
            if (sys.argv[i+1] == "1" or sys.argv[i+1] == "2" or sys.argv[i+1] == "3" or sys.argv[i+1] == "4"):
                mode = sys.argv[i+1]
            else:
                print("Mode number should be either 1 or 2 or 3 or 4")
                exit(1)
    return int(mode)


def padImage(image):
    result = np.zeros((next_power_of_2(
        image.shape[0]), next_power_of_2(image.shape[1])))
    result[:image.shape[0], :image.shape[1]] = image
    return result


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def denoiseImage(image):
    img = fft_2D(padImage(image))
    p = 0.10
    r = len(img)
    c = len(img[0])
    img[int(r*p):int(r*(1-p))] = 0
    img[:, int(c*p):int(c*(1-p))] = 0
    return np.abs(ifft_2D(img))


def compress(image, per):
    img = fft_2D(padImage(image))
    r, c = image.shape
    p = (1-per)/4
    img[int(r*p):int(r*(1-p))] = 0
    img[:, int(c*p):int(c*(1-p))] = 0
    print("Number of non-zero elements in %" +
          str(per*100) + " compressed FFT:")
    print(np.count_nonzero(img))
    # Save image
    outputName = str(int(per*100)) + "%CompressedFFT"
    s_img = sparse.csr_matrix(img)
    sparse.save_npz(outputName, s_img)
    return np.abs(ifft_2D(img))


def crop(originalImage, compressedImage):
    image = compressedImage[0:len(
        originalImage), 0:len(originalImage[0])]
    return image


def Average(lst):
    return sum(lst) / len(lst)


def plotTC_1D(f_2d, type):
    aName = "Naive Discrete Fourier Transform" if type == "naive" else "Fast Fourier Transform"
    x = []
    y = []
    for i in [2**6, 2**8]:
        doubleArray = np.random.random((i, i))
        timeArray = []
        for k in range(1, 10):
            start = time.perf_counter()
            f_2d(doubleArray)
            end = time.perf_counter()
            timeArray.append(end-start)
        x.append(i)
        y.append(Average(timeArray))
        print("The avarage time it takes " + aName +
              " algorithm given " + str(i) + " input size is: ")
        print(Average(timeArray))
        print("The variance of " + aName +
              " algorithm given " + str(i) + " input size is: ")
        print(np.var(timeArray))
    p1 = plt.plot(x, y, label=type)
    plt.xlabel('Problem Size')
    plt.ylabel('Seconds')
    plt.legend()


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

def naiveDFT_2D(a):
    x = np.zeros(a.shape, dtype=complex)
    M, N = a.shape
    for i in range(N):
        x[:, i] = naiveDFT_1D(a[:, i])
    for i in range(M):
        x[i, :] = naiveDFT_1D(x[i, :])
    return x


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


def modeThree():
    # create a template for 2 images
    f = plt.figure()
    f.add_subplot(2, 3, 1)
    # add original image
    originalImage = plt.imread(getImageString()).astype(float)
    plt.title('Original Image')
    plt.imshow(originalImage, cmap="gray")
    compress(originalImage, 0)
    # add image
    f.add_subplot(2, 3, 2)
    plt.title('%20 Trunctated')
    compressedImage1 = compress(originalImage, 0.2)
    image1 = crop(originalImage, compressedImage1)
    plt.imshow(image1, cmap='gray')
    # add image
    f.add_subplot(2, 3, 3)
    plt.title('%44 Trunctated')
    compressedImage2 = compress(originalImage, 0.44)
    image2 = crop(originalImage, compressedImage2)
    plt.imshow(image2, cmap='gray')
    # add image
    f.add_subplot(2, 3, 4)
    plt.title('%58 Trunctated')
    compressedImage3 = compress(originalImage, 0.58)
    image3 = crop(originalImage, compressedImage3)
    plt.imshow(image3, cmap='gray')
    # add image
    f.add_subplot(2, 3, 5)
    plt.title('%76 Trunctated')
    comressedImage4 = compress(originalImage, 0.76)
    image4 = crop(originalImage, comressedImage4)
    plt.imshow(image4, cmap='gray')
    # add image
    f.add_subplot(2, 3, 6)
    plt.title('%95 Trunctated')
    compressedImage5 = compress(originalImage, 0.95)
    image5 = crop(originalImage, compressedImage5)
    plt.imshow(image5, cmap='gray')
    # print image
    plt.show(block=True)


def modeFour():
    plotTC_1D(naiveDFT_2D, "naive")
    plotTC_1D(fft_2D, "fast")
    plt.show()


# Main

def main():
    mode = getModeNumber()
    if(mode == 1):
        modeOne()
    elif(mode == 2):
        modeTwo()
    elif(mode == 3):
        modeThree()
    elif(mode == 4):
        modeFour()
    else:
        modeOne()


main()
