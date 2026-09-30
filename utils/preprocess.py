import json
import os

import cv2
import numpy as np

from utils.general import form_contours
from utils.image import color_threshold
from utils.settings import (
    ARTIFACT_LOCATIONS,
    DEBUG,
    MANUAL_WINDOW_NAME,
    MAX_BEE_AREA,
    MAX_GROUP_AREA,
    MIN_BEE_AREA,
    WINDOW_NAME,
)


NUM_BEES = 0


LEFT_ARROW_KEYS = {2, 81, 2424832, 65361, 63234}
RIGHT_ARROW_KEYS = {3, 83, 2555904, 65363, 63235}


def _setting_adjustment(key, setting):
    if key in LEFT_ARROW_KEYS:
        direction = -1
    elif key in RIGHT_ARROW_KEYS:
        direction = 1
    else:
        return None

    step = 0.5 if setting == "color_multiplier" else 1
    return direction * step


def close_auxiliary_windows():
    window_names = {
        "img",
        "Contour preview",
        "Preprocessing settings",
        MANUAL_WINDOW_NAME,
    }
    window_names.discard(WINDOW_NAME)
    for window_name in window_names:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


def load_settings(src_processed_root, frame_num, default=None):
    settings_path = os.path.join(src_processed_root, "preprocess_settings.json")
    if not os.path.exists(settings_path):
        print(f"No settings found for frame: {frame_num}")
        return default

    with open(settings_path, encoding="utf-8") as settings_file:
        saved_settings = json.load(settings_file)
    settings = saved_settings.get(str(frame_num), default)
    if settings is None:
        print(f"No settings found for frame: {frame_num}")
        return default
    print("Loaded settings: ", settings)
    return settings


def save_settings(src_processed_root, frame_num, settings):
    settings_path = os.path.join(src_processed_root, "preprocess_settings.json")
    if os.path.exists(settings_path):
        with open(settings_path, "r", encoding="utf-8") as settings_file:
            saved_settings = json.load(settings_file)
    else:
        saved_settings = {}
    saved_settings[str(frame_num)] = settings
    with open(settings_path, "w", encoding="utf-8") as settings_file:
        json.dump(saved_settings, settings_file)


def default_preprocess_settings(frame_num):
    return {
        "frame_num": frame_num,
        "color_multiplier": 7,
        "thresh": 120,
        "color_thresh": 1,
        "dilate_iter": 8,
        "erode_iter": 2,
        "artifacts": ARTIFACT_LOCATIONS,
        "num_bees": 0,
        "bee_colors": [],
    }


def preprocess(frame, frame_shape, settings):
    resized_frame = cv2.resize(frame, frame_shape)
    grayscale = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    multiplied = cv2.multiply(grayscale, np.full(resized_frame.shape[-1], settings["color_multiplier"]))
    multiplied = cv2.cvtColor(multiplied, cv2.COLOR_GRAY2BGR)

    color_mask = np.zeros(resized_frame.shape[:2], dtype=np.uint8)
    for bee_color in settings["bee_colors"]:
        color_mask = cv2.bitwise_or(color_mask, color_threshold(resized_frame, bee_color, settings["color_thresh"]))
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
    background_sub = cv2.bitwise_and(multiplied, color_mask).astype(np.uint8)
    background_sub = cv2.cvtColor(background_sub, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(background_sub, (25, 25), 0)
    _, threshold = cv2.threshold(blurred, settings["thresh"], 255, cv2.THRESH_BINARY)
    eroded = cv2.erode(threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=settings["erode_iter"])
    dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=settings["dilate_iter"])

    for artifact in settings["artifacts"]:
        cv2.rectangle(dilated, artifact, 0, -1)
    return dilated, background_sub


def _preview_images(frame, settings, previous_contours=None):
    global NUM_BEES
    NUM_BEES = settings["num_bees"]
    resized_frame = cv2.resize(frame, (frame.shape[1], frame.shape[0]))
    image, background_sub, threshold, no_artifacts, frame_no_artifacts = update_frame(
        resized_frame,
        settings,
        previous_contours=previous_contours,
    )
    result = np.concatenate(
        (np.concatenate((image, background_sub), axis=1), np.concatenate((no_artifacts, frame_no_artifacts), axis=1)),
        axis=0,
    )
    cv2.imshow("Contour preview", frame_no_artifacts)
    return result


def _show_preprocess_preview(preview, current_setting, value):
    print(f"Editing {current_setting}={value}")
    cv2.imshow("Preprocessing settings", preview)


def edit_preprocess_settings(
    frame_num,
    settings,
    frame,
    src_processed_root,
    keep_preview=False,
    previous_contours=None,
):
    revised_settings = dict(settings)
    current_setting = "thresh"
    preview = _preview_images(frame, revised_settings, previous_contours)
    _show_preprocess_preview(preview, current_setting, revised_settings[current_setting])

    while True:
        key = cv2.waitKeyEx(30)
        amount = _setting_adjustment(key, current_setting)
        if amount is not None:
            revised_settings[current_setting] += amount
            if current_setting in ("dilate_iter", "erode_iter"):
                revised_settings[current_setting] = max(0, revised_settings[current_setting])
            print(f"{current_setting}=", revised_settings[current_setting])
            preview = _preview_images(frame, revised_settings, previous_contours)
            _show_preprocess_preview(preview, current_setting, revised_settings[current_setting])
        elif key in (ord("m"), ord("t"), ord("c"), ord("d"), ord("e")):
            current_setting = {
                ord("m"): "color_multiplier",
                ord("t"): "thresh",
                ord("c"): "color_thresh",
                ord("d"): "dilate_iter",
                ord("e"): "erode_iter",
            }[key]
            print(f"modifying {current_setting}")
            _show_preprocess_preview(preview, current_setting, revised_settings[current_setting])
        elif key in (13, 27):
            break

    if not keep_preview:
        close_auxiliary_windows()
    # Keep the preview visible during manual segmentation
    save_settings(src_processed_root, frame_num, revised_settings)
    return revised_settings


def set_preprocess_settings(frame_num, settings, cap, frame, frame_shape, src_processed_root):
    global NUM_BEES

    NUM_BEES = settings["num_bees"]
    resized_frame = cv2.resize(frame, frame_shape)
    preview = _preview_images(resized_frame, settings)
    _show_preprocess_preview(preview, "thresh", settings["thresh"])

    current_setting = "thresh"
    while cap.isOpened():
        key = cv2.waitKeyEx(30)
        amount = _setting_adjustment(key, current_setting)
        if amount is not None:
            settings[current_setting] += amount
            if current_setting in ("dilate_iter", "erode_iter"):
                settings[current_setting] = max(0, settings[current_setting])
            print(f"{current_setting}=", settings[current_setting])
            preview = _preview_images(resized_frame, settings)
            _show_preprocess_preview(preview, current_setting, settings[current_setting])
        elif key in (ord("m"), ord("t"), ord("c"), ord("d"), ord("e")):
            current_setting = {
                ord("m"): "color_multiplier",
                ord("t"): "thresh",
                ord("c"): "color_thresh",
                ord("d"): "dilate_iter",
                ord("e"): "erode_iter",
            }[key]
            print(f"modifying {current_setting}")
            _show_preprocess_preview(preview, current_setting, settings[current_setting])
        elif key in (13, 27):
            break

    close_auxiliary_windows()
    save_settings(src_processed_root, frame_num, settings)


def update_frame(frame, settings, previous_contours=None):
    image = cv2.multiply(frame, np.full(frame.shape[-1], settings["color_multiplier"]))
    color_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for bee_color in settings["bee_colors"]:
        color_mask = cv2.bitwise_or(color_mask, color_threshold(frame, bee_color, settings["color_thresh"]))
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
    background_sub = cv2.bitwise_and(image, color_mask).astype(np.uint8)
    background_sub = cv2.cvtColor(background_sub, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(background_sub, (25, 25), 0)
    _, threshold = cv2.threshold(blurred, settings["thresh"], 255, cv2.THRESH_BINARY)
    eroded = cv2.erode(threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=settings["erode_iter"])
    dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=settings["dilate_iter"])

    background_sub = cv2.cvtColor(background_sub, cv2.COLOR_GRAY2BGR)
    threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("dilate", frame)
    no_artifacts = np.copy(dilated)
    frame_no_artifacts = np.copy(frame)
    for artifact in settings["artifacts"]:
        no_artifacts = cv2.rectangle(no_artifacts, artifact, 0, -1)
        frame_no_artifacts = cv2.rectangle(frame_no_artifacts, artifact, (0, 0, 255), -1)

    contours, distant_contours = form_contours(
        no_artifacts,
        MIN_BEE_AREA,
        MAX_GROUP_AREA,
        NUM_BEES=settings["num_bees"],
        remove_background=False,
        remove_extra_contours=False,
        previous_contours=previous_contours,
        return_distance_rejected=True,
    )
    cv2.drawContours(frame_no_artifacts, distant_contours, -1, (255, 0, 255), 2)
    cv2.drawContours(frame_no_artifacts, contours, -1, (0, 255, 0), 2)
    for contour in contours:
        x, y, _, height = cv2.boundingRect(contour)
        area_label = f"Area: {cv2.contourArea(contour):.0f}"
        label_y = y - 6 if y > 20 else y + height + 16
        label_origin = (x, label_y)
        cv2.putText(
            frame_no_artifacts,
            area_label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_no_artifacts,
            area_label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    if contours and DEBUG:
        areas = [cv2.contourArea(contour) for contour in contours]
        print("min area", min(areas))
        print("max area", max(areas))
    no_artifacts = cv2.cvtColor(no_artifacts, cv2.COLOR_GRAY2BGR)
    return image, background_sub, threshold, no_artifacts, frame_no_artifacts
