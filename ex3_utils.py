import sys
from typing import List
import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[y,x]...], [[dU,dV]...] for each points
    """
    if win_size % 2 == 0:
        win_size = win_size - 1

    x = cv2.Sobel(im2, cv2.CV_64F, 1, 0, ksize=win_size)
    y = cv2.Sobel(im2, cv2.CV_64F, 0, 1, ksize=win_size)
    temp = im2 - im1
    mat = []
    dots = []
    for i in range(win_size, im1.shape[0] - win_size + 1, step_size):
        for j in range(win_size, im1.shape[1] - win_size + 1, step_size):
            xi = i - win_size // 2
            xj = j - win_size // 2
            yi = i + win_size // 2 + 1
            yj = j + win_size // 2 + 1
            a = -(temp[xi:yi, xj:yj]).reshape(win_size ** 2, 1)
            b = np.asmatrix(np.concatenate((x[xi:yi, xj:yj].reshape(win_size ** 2, 1), y[xi:yi, xj:yj].reshape(win_size ** 2, 1)),axis=1))
            (val, vec) = np.linalg.eig(b.T * b)
            val.sort()
            val = val[::-1]
            if val[0] >= val[1] > 1 and val[0] / val[1] < 100:
                temp = np.array(np.dot(np.linalg.pinv(b), a))
                mat.append(np.array([j, i]))
                dots.append(temp[::-1].copy())
    return np.array(mat), np.array(dots)


def getkernel() -> np.ndarray:
    kernel = cv2.getGaussianKernel(5, 1.1)
    kernel = kernel * kernel.T
    kernel /= kernel.sum()

    return kernel


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    im = [img]
    kernel = getkernel()
    laplaceian = []
    for i in range(1, levels):
        temp = cv2.filter2D(im[i - 1], -1, kernel)
        laplaceian.append(im[i - 1] - temp)
        im.append(temp[::2, ::2])
    laplaceian.append(im[-1])
    return laplaceian


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    gaussian = [lap_pyr[-1]]
    kernel = getkernel()
    laplaceian = lap_pyr.copy()
    laplaceian.reverse()
    for i in range(1, len(lap_pyr)):
        gaussian.append(laplaceian[i] + gaussExpand(gaussian[i - 1], kernel))
    return gaussian[-1]


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    kernel = getkernel()
    gaussianPyr = [img]
    for i in range(1, levels):
        temp = cv2.filter2D(gaussianPyr[i - 1], -1, kernel)
        gaussianPyr.append(temp[::2, ::2].copy())
    return gaussianPyr


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    if (len(img.shape) == 2):
        weight = img.shape[0]
        hight = img.shape[1]
        ans = np.full((2 * weight, 2 * hight), 0, dtype=img.dtype)
        ans = ans.astype(np.float)
        ans[::2, ::2] = img
    if (len(img.shape) == 3):
        (weight, hight, depth) = img.shape
        ans = np.full((2 * weight, 2 * hight, depth), 0, dtype=img.dtype)
        ans = ans.astype(np.float)
        ans[::2, ::2] = img

    kernel = (gs_k * 4) / gs_k.sum()
    ans = cv2.filter2D(ans, -1, kernel, borderType=cv2.BORDER_DEFAULT)

    return ans


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: Blended Image
    """
    blend1 = img_1 * mask + img_2 * (1 - mask)
    a = laplaceianReduce(img_1, levels)
    b = laplaceianReduce(img_2, levels)
    gaussianP = gaussianPyr(mask, levels)

    temp = []
    for i in range(levels):
        temp.append(gaussianP[i] * a[i] + (1 - gaussianP[i]) * b[i])
    blend2 = laplaceianExpand(temp)
    return (blend1, blend2)