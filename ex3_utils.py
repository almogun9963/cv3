import sys
from typing import List
import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    cols = im1.width
    rows = im1.height

    velx = cv2.CreateMat(rows, cols, cv2.CV_32FC1)
    vely = cv2.CreateMat(rows, cols, cv2.CV_32FC1)

    ans1 =cv2.CalcOpticalFlowHS(im1, im2, False, velx, vely, 100.0,(cv2.CV_TERMCRIT_ITER | cv2.CV_TERMCRIT_EPS, 64, 0.01))

    ans2 =cv2.CalcOpticalFlowLK(im1, im2, (win_size, win_size), velx, vely)

    return (ans1,ans2)


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

    LS = []
    for i in range(levels):
        LS.append(kernel[i] * im1_p[i] + (1 - kernel[i]) * im2_p[i])

    blended = laplaceianExpand(LS)
    return (naive, blended)