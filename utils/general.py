import glob
import math
import os
import shutil

import cv2
import numpy as np

from utils.settings import DEBUG, MAX_CONTOUR_DISTANCE_RATIO


def filter_contours(
    contours,
    min_area,
    max_area,
    max_count=None,
    previous_contours=None,
    max_distance=None,
    return_distance_rejected=False,
):
    """Filter contours by area, proximity to prior contours, and optional count."""
    previous_count = len(contours)
    contours = [
        contour
        for contour in contours
        if min_area < cv2.contourArea(contour) < max_area
    ]
    if previous_count != len(contours) and DEBUG:
        print("Warning: Removed contours based on area")

    previous_centers = [
        _contour_center(contour)
        for contour in (previous_contours or [])
    ]
    distance_rejected = []
    if previous_centers and max_distance is not None:
        nearby_contours = []
        for contour in contours:
            center = _contour_center(contour)
            nearest_distance = min(
                math.hypot(
                    center[0] - reference_point[0],
                    center[1] - reference_point[1],
                )
                for reference_point in previous_centers
            )
            if nearest_distance <= max_distance:
                nearby_contours.append(contour)
            else:
                distance_rejected.append(contour)
        if len(nearby_contours) != len(contours) and DEBUG:
            print("Warning: Removed contours too far from previous tracks")
        contours = nearby_contours

    if max_count is not None and len(contours) > max_count:
        excess = len(contours) - max_count
        contours = sorted(contours, key=cv2.contourArea)[excess:]
        if DEBUG:
            print("Warning: Removed excess contours based on NUM_BEES")

    if return_distance_rejected:
        return contours, distance_rejected
    return contours


def _contour_center(contour):
    moments = cv2.moments(contour)
    if moments["m00"]:
        return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    x, y, width, height = cv2.boundingRect(contour)
    return x + width / 2, y + height / 2


def form_contours(
    image,
    minArea,
    maxArea,
    NUM_BEES,
    remove_background=False,
    remove_extra_contours=False,
    previous_contours=None,
    return_distance_rejected=False,
):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return (contours, []) if return_distance_rejected else contours

    hierarchy = hierarchy[0]
    if remove_background:
        contours = [contour for index, contour in enumerate(contours) if hierarchy[index][2] == -1]

    max_count = NUM_BEES if remove_extra_contours else None
    max_distance = math.hypot(image.shape[1], image.shape[0]) * MAX_CONTOUR_DISTANCE_RATIO
    return filter_contours(
        contours,
        minArea,
        maxArea,
        max_count=max_count,
        previous_contours=previous_contours,
        max_distance=max_distance,
        return_distance_rejected=return_distance_rejected,
    )


def imgs2vid(images, output_path, fps):
    height, width = images[0].shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc("m", "p", "4", "v"), fps, (width, height), True)
    for image in images:
        writer.write(image)
    cv2.destroyAllWindows()
    writer.release()


def setup_draw_img(base_img):
    draw_img = base_img.copy()
    if len(draw_img.shape) == 2:
        draw_img = np.tile(draw_img[..., np.newaxis], (1, 1, 3))
    elif draw_img.shape[-1] == 1:
        draw_img = np.tile(draw_img, (1, 1, 3))
    return draw_img


def compute_centroid(x, y, width, height):
    return int(x + width / 2), int(y + height / 2)


def log_data(data_logger, x, y, width, height, group):
    if data_logger is not None:
        data_logger.append({"x": float(x), "y": float(y), "h": float(height), "w": float(width), "id": group})
    return data_logger


def make_directory(dirname, root=os.getcwd(), remove_old=False):
    new_directory = os.path.join(root, dirname.replace("/", os.path.sep))
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    elif remove_old:
        shutil.rmtree(new_directory)
        os.makedirs(new_directory)
    return new_directory


def select_file(source_root, max_failed_attempts=3, prefix=""):
    paths = sorted(glob.glob(f"{source_root}/*{prefix}"))
    for index, path in enumerate(paths):
        print(f"{index} : {os.path.basename(path)}")

    for _ in range(max_failed_attempts):
        try:
            choice = int(input("Select video by index: "))
        except ValueError:
            print("Enter valid index integer")
            continue
        if 0 <= choice < len(paths):
            return paths[choice]
        print(f"Invalid index choice. Only {len(paths)} entries exist to choose from.")

    raise SystemExit("Too many failed attempts. Exiting program.")
