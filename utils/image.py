import cv2
import numpy as np


def color_threshold(image, color, offset):
    upper_color = (np.array(color) + offset).astype(image.dtype)
    return cv2.inRange(image, np.zeros(len(upper_color), dtype=image.dtype), upper_color)


def adaptive_thresholding(img, invert=True):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, img_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert:
        img_mask = 255 - img_mask
    return img_mask


def adaptive_filter_plus_opening(img, kernel_dim=(9, 9), invert=False):
    img_mask = adaptive_thresholding(img, invert=invert)
    kernel = np.ones(kernel_dim, np.uint8)
    return cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel)


def draw_box(draw_img, x, y, w, h, box_width, color, draw_centroid=True):
    cv2.rectangle(draw_img, (x, y), (x + w, y + h), color, box_width)
    if draw_centroid:
        cv2.circle(draw_img, (int(x + w / 2), int(y + h / 2)), 5, color, -1)


def remove_lines(gray):
    rowvals = np.mean(np.sort(gray, axis=1)[:, -100:], axis=1)
    graymod = np.copy(gray).astype(float)
    graymod *= np.expand_dims(np.max(rowvals) / np.array(rowvals), axis=1)
    return graymod.astype(np.uint8)


def remove_lines_old(gray):
    rowvals = np.mean(np.sort(gray, axis=1)[:, -100:], axis=1)
    graymod = np.copy(gray).astype(float)
    graymod *= np.expand_dims(np.max(rowvals) / np.array(rowvals), axis=1)
    graymod = np.clip(graymod, 0, 255)
    return graymod.astype(np.uint32)


def remove_lines_BGR(img_BGR):
    img_BGR_no_lines = np.zeros_like(img_BGR)
    for channel_index in range(3):
        img_BGR_no_lines[:, :, channel_index] = remove_lines_old(img_BGR[:, :, channel_index])
    return img_BGR_no_lines
