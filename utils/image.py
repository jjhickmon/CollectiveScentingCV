import cv2
import numpy as np
import utils.general as general_utils

def adaptive_thresholding(img, invert=True):
    blur = cv2.GaussianBlur(img,(5,5),0)
    _, img_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    if invert:
        img_mask = 255 - img_mask

    return img_mask

def adaptive_filter_plus_opening(img, kernel_dim=(9,9), invert=False):
    img_mask = adaptive_thresholding(img, invert=invert)

    #----- For trophallaxis: try eroding first
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6)) #np.ones((5,5),np.uint8)
    # img_mask = cv2.erode(img_mask, kernel, iterations=2)
    # kernel = np.ones(kernel_dim, np.uint8)
    # img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)
    #--------------------------------------------------#

    # Opening morphology
    kernel = np.ones(kernel_dim, np.uint8)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel)

    return img_mask

def draw_box(draw_img, x,y,w,h, box_width, color, draw_centroid=True):
    top_left = (x,y)
    bottom_right = (x+w, y+h)
    cv2.rectangle(draw_img, top_left, bottom_right, color, box_width)
    if draw_centroid:
        centroid = general_utils.compute_centroid(x,y,w,h)
        cv2.circle(draw_img, centroid, 5, color, -1)

def remove_lines(gray):
    rowvals = np.mean(np.sort(gray, axis=1)[:,-100:], axis=1)
    graymod = np.copy(gray).astype(np.float)
    graymod *= np.expand_dims(np.max(rowvals) / np.array(rowvals), axis=1)
    return graymod.astype(np.uint8)
