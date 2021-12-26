import numpy as np
import cv2
from skimage.color import rgb2hsv, hsv2rgb


def RecoverHE(sceneRadiance):
    for i in range(3):
        sceneRadiance[:, :, i] = cv2.equalizeHist(sceneRadiance[:, :, i])
    return sceneRadiance


def RecoverCLAHE(sceneRadiance):
    clahe = cv2.createCLAHE(clipLimit=7, tileGridSize=(14, 14))
    for i in range(3):
        sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))
    return sceneRadiance


def RecoverGC(sceneRadiance):
    sceneRadiance = sceneRadiance / 255.0
    for i in range(3):
        sceneRadiance[:, :, i] = np.power(sceneRadiance[:, :, i] / float(np.max(sceneRadiance[:, :, i])), 3.2)
    sceneRadiance = np.clip(sceneRadiance * 255, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance


def _stretching(img):
    height = len(img)
    width = len(img[0])
    for k in range(0, 3):
        Max_channel = np.max(img[:, :, k])
        Min_channel = np.min(img[:, :, k])
        for i in range(height):
            for j in range(width):
                img[i, j, k] = (img[i, j, k] - Min_channel) * (255 - 0) / (Max_channel - Min_channel) + 0
    return img


def _global_stretching(img_L, height, width):
    I_min = np.min(img_L)
    I_max = np.max(img_L)
    array_Global_histogram_stretching_L = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            p_out = (img_L[i][j] - I_min) * ((1) / (I_max - I_min))
            array_Global_histogram_stretching_L[i][j] = p_out
    return array_Global_histogram_stretching_L


def _HSVStretching(sceneRadiance):
    height = len(sceneRadiance)
    width = len(sceneRadiance[0])
    img_hsv = rgb2hsv(sceneRadiance)
    h, s, v = cv2.split(img_hsv)
    img_s_stretching = _global_stretching(s, height, width)
    img_v_stretching = _global_stretching(v, height, width)
    labArray = np.zeros((height, width, 3), "float64")
    labArray[:, :, 0] = h
    labArray[:, :, 1] = img_s_stretching
    labArray[:, :, 2] = img_v_stretching
    img_rgb = hsv2rgb(labArray) * 255


def _sceneRadianceRGB(sceneRadiance):
    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance


def RecoverICM(img1):
    img = _stretching(img1)
    sceneRadiance = _sceneRadianceRGB(img)
    sceneRadiance = _HSVStretching(sceneRadiance)
    sceneRadiance = _sceneRadianceRGB(sceneRadiance)
    return sceneRadiance
