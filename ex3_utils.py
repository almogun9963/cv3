import sys
from typing import List
import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    if win_size % 2 == 0:
        win_size = win_size + 1

    x = cv2.Sobel(im2, cv2.CV_64F, 1, 0, ksize=5)
    y = cv2.Sobel(im2, cv2.CV_64F, 0, 1, ksize=5)
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
            b = np.asmatrix(np.concatenate((x[xi:yi, xj:yj].reshape(win_size ** 2, 1),y[xi:yi, xj:yj].reshape(win_size ** 2, 1)), axis=1))
            v, vector = np.linalg.eig(b.T * b)
            v.sort()
            v = v[::-1]
            if  v[0] / v[1] < 100 and v[0] >= v[1] > 1 :
                t = np.array(np.dot(np.linalg.pinv(b), a))
                mat.append(np.array([j, i]))
                dots.append(t[::-1].copy())
    return np.array(mat), np.array(dots)


def getkernel() -> np.ndarray:
    kernel = cv2.getGaussianKernel(5, 1.1)
    kernel = kernel * kernel.T
    kernel /= kernel.sum()

    return kernel


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
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
    gaussian = [lap_pyr[-1]]
    kernel = getkernel()
    laplaceian = lap_pyr.copy()
    laplaceian.reverse()
    for i in range(1, len(lap_pyr)):
        gaussian.append(laplaceian[i] + gaussExpand(gaussian[i - 1], kernel))
    return gaussian[-1]


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    kernel = getkernel()
    gaussianPyr = [img]
    for i in range(1, levels):
        temp = cv2.filter2D(gaussianPyr[i - 1], -1, kernel)
        gaussianPyr.append(temp[::2, ::2].copy())
    return gaussianPyr


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
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
    kernel = getkernel()
    naive = img_1 * mask + img_2 * (1 - mask)
    im1_p = laplaceianReduce(img_1, levels)
    im2_p = laplaceianReduce(img_2, levels)
    gaussianP = gaussianPyr(mask, levels)

    temp = []
    for i in range(levels):
        temp.append(gaussianP[i] * im1_p[i] + (1 - gaussianP[i]) * im2_p[i])


    blended = laplaceianExpand(temp)
    return (naive, blended)