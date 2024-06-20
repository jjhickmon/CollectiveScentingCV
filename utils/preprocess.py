import cv2
import numpy as np
import json
from utils.general import form_contours
from utils.image import color_threshold
from utils.tracker import manual_segmentation
from utils.settings import *

def preprocess(frame, frame_shape, color_multiplier, thresh, color_thresh_val, bee_colors, dilate_iter, erode_iter, artifacts):
    #set img and background to same size
    frame = cv2.resize(frame, frame_shape)
    bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.multiply(bw, np.full(frame.shape[-1], color_multiplier))
    cv2.imshow("img", img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # img_background = cv2.multiply(background, np.array([color_multiplier]))

    # Background Subtraction
    # background_sub = cv2.cvtColor(cv2.absdiff(img, img_background), cv2.COLOR_BGR2GRAY)
    # cv2.imshow("bg", background_sub)
    src_img = np.copy(frame)
    # Threshold based on the colors of the points
    color_thresh = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    for bee_color in bee_colors:
        color_thresh = cv2.bitwise_or(color_thresh, color_threshold(src_img, bee_color, color_thresh_val))
    color_thresh = cv2.cvtColor(color_thresh, cv2.COLOR_GRAY2BGR)
    background_sub = cv2.bitwise_and(img, color_thresh).astype(np.uint8)
    background_sub = cv2.cvtColor(background_sub, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(background_sub, (25, 25), 0) # blur to quickly denoise, precision is not important for this step
    _, threshold = cv2.threshold(blurred, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)
    erode = cv2.erode(threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=erode_iter)
    dilate = cv2.dilate(erode, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=dilate_iter)

    for artifact in artifacts:
        cv2.rectangle(dilate, artifact, 0, -1)
    cv2.imshow("thresh", dilate)
    return dilate, background_sub


def set_preprocess_settings(cap, frame, frame_shape, src_processed_root):
    global NUM_BEES

    settings = {}
    settings["color_multiplier"] = 7
    settings["thresh"] = 120
    settings["color_thresh"] = 1
    settings["dilate_iter"] = 8
    settings["erode_iter"] = 2
    # NOTE: queen's cage, bottm left, top right
    settings["artifacts"] = ARTIFACT_LOCATIONS #[(485, 35, 95, 385), (410, 1000, 50, 100), (1200, 0, 300, 30)]
    settings["num_bees"] = 0

    cv2.imshow("frame", frame)
    custom_points, custom_mask = manual_segmentation(frame, np.full(frame.shape, 0), np.full((frame.shape[0], frame.shape[1], 1), 255), None)
    settings["num_bees"] = len(custom_points)
    NUM_BEES = settings["num_bees"]

    point_colors = []
    for point in custom_points:
        point_colors.append(frame[point[1], point[0]].tolist())
    settings["bee_colors"] = point_colors

    curr_setting = "color_multiplier"
    frame = cv2.resize(frame, frame_shape)
    img, background_sub, threshold, no_artifacts, frame_no_artifacts = update_frame(frame, settings)
    # NOTE: top-left is img, top-right is background_sub, bottom-left is no_artifacts, bottom-right is frame_no_artifacts
    result = np.concatenate((np.concatenate((img, background_sub), axis=1), np.concatenate((no_artifacts, frame_no_artifacts), axis=1)), axis=0)
    title = f"modifying settings {curr_setting}={settings[curr_setting]}"
    cv2.imshow(title, result)
    while (cap.isOpened()):
        k = cv2.waitKey(30) & 0xff
        if k == 2: # left arrow
            if curr_setting == "color_multiplier":
                settings["color_multiplier"] -= 0.5
                print("color_multiplier=",settings["color_multiplier"])
            if curr_setting == "thresh":
                settings["thresh"] -= 1
                print("thresh=",settings["thresh"])
            if curr_setting == "color_thresh":
                settings["color_thresh"] -= 1
                print("color_thresh=",settings["color_thresh"])
            cv2.destroyAllWindows()
            img, background_sub, threshold, no_artifacts, frame_no_artifacts = update_frame(frame, settings)
            result = np.concatenate((np.concatenate((img, background_sub), axis=1), np.concatenate((no_artifacts, frame_no_artifacts), axis=1)), axis=0)
            title = f"modifying settings {curr_setting}={settings[curr_setting]}"
            cv2.imshow(title, result)
        elif k == 3: # right arrow
            if curr_setting == "color_multiplier":
                settings["color_multiplier"] += 0.5
                print("color_multiplier=",settings["color_multiplier"])
            if curr_setting == "thresh":
                settings["thresh"] += 1
                print("thresh=",settings["thresh"])
            if curr_setting == "color_thresh":
                settings["color_thresh"] += 1
                print("color_thresh=",settings["color_thresh"])
            cv2.destroyAllWindows()
            img, background_sub, threshold, no_artifacts, frame_no_artifacts = update_frame(frame, settings)
            result = np.concatenate((np.concatenate((img, background_sub), axis=1), np.concatenate((no_artifacts, frame_no_artifacts), axis=1)), axis=0)
            title = f"modifying settings {curr_setting}={settings[curr_setting]}"
            cv2.imshow(title, result)
        elif k == ord('m'): # m key
            curr_setting = "color_multiplier"
            print("modifying color_multiplier")
            cv2.destroyAllWindows()
            cv2.imshow(f"modifying settings {curr_setting}={settings[curr_setting]}", result)
        elif k == ord('t'): # t key
            curr_setting = "thresh"
            print("modifying thresh")
            cv2.destroyAllWindows()
            cv2.imshow(f"modifying settings {curr_setting}={settings[curr_setting]}", result)
        elif k == ord('c'): # c key
            curr_setting = "color_thresh"
            print("modifying color_thresh")
            cv2.destroyAllWindows()
            cv2.imshow(f"modifying settings {curr_setting}={settings[curr_setting]}", result)
        if k == 27 or k == 13: # escape key or enter
            break

    cv2.destroyAllWindows()
    with open(f"{src_processed_root}/preprocess_settings.json", "w") as outfile:
        json.dump(settings, outfile)

def update_frame(frame, settings):
    color_multiplier = settings["color_multiplier"]
    thresh = settings["thresh"]
    color_thresh_val = settings["color_thresh"]
    bee_colors = settings["bee_colors"]
    dilate_iter = settings["dilate_iter"]
    erode_iter = settings["erode_iter"]
    artifacts = settings["artifacts"]

    img = cv2.multiply(frame, np.full(frame.shape[-1], color_multiplier))
    # img_background = cv2.multiply(background, np.full(background.shape[-1], color_multiplier))
    # background_sub = cv2.cvtColor(cv2.absdiff(img, img_background), cv2.COLOR_BGR2GRAY)

    src_img = np.copy(frame)
    # Threshold based on the colors of the points
    color_thresh = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    for bee_color in bee_colors:
        color_thresh = cv2.bitwise_or(color_thresh, color_threshold(src_img, bee_color, color_thresh_val))
    color_thresh = cv2.cvtColor(color_thresh, cv2.COLOR_GRAY2BGR)
    background_sub = cv2.bitwise_and(img, color_thresh).astype(np.uint8)
    background_sub = cv2.cvtColor(background_sub, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(background_sub, (25, 25), 0) # blur to quickly denoise, precision is not important for this step
    _, threshold = cv2.threshold(blurred, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)
    erode = cv2.erode(threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=erode_iter)
    dilate = cv2.dilate(erode, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=dilate_iter)

    background_sub = cv2.cvtColor(background_sub, cv2.COLOR_GRAY2BGR)
    threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
    cv2.imshow("dilate", frame)
    no_artifacts = np.copy(dilate)
    color_no_artifacts = np.copy(frame)
    for artifact in artifacts:
        no_artifacts = cv2.rectangle(no_artifacts, artifact, 0, -1)
        color_no_artifacts = cv2.rectangle(color_no_artifacts, artifact, (0, 0, 255), -1)

    contours = form_contours(no_artifacts, MIN_BEE_AREA, MAX_GROUP_AREA, NUM_BEES=NUM_BEES, remove_background=True, remove_extra_contours=True)
    contours2 = form_contours(no_artifacts, 0, 100000, NUM_BEES=NUM_BEES, remove_background=False, remove_extra_contours=False)
    cv2.drawContours(color_no_artifacts, contours2, -1, (0, 0, 255), 2)
    cv2.drawContours(color_no_artifacts, contours, -1, (0, 255, 0), 2)
    # print min area and max area detected from contours 2
    if len(contours2) > 0:
        print("min area", min([cv2.contourArea(contour) for contour in contours2]))
        print("max area", max([cv2.contourArea(contour) for contour in contours2]))
    no_artifacts = cv2.cvtColor(no_artifacts, cv2.COLOR_GRAY2BGR)
    return img, background_sub, threshold, no_artifacts, color_no_artifacts