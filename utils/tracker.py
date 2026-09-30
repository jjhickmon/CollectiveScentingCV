import cv2
import numpy as np

from utils.image import color_threshold
from utils.preprocess import close_auxiliary_windows, edit_preprocess_settings, load_settings, preprocess
from utils.settings import (
    ALLOW_MANUAL_SEGMENTING,
    MAX_AUTOMATIC_ITERATIONS,
    MAX_BEE_AREA,
    MAX_GROUP_AREA,
    MAX_THRESH_COLOR_DIFF,
    MIN_BEE_AREA,
    MANUAL_WINDOW_NAME,
    DEBUG,
)
from utils.general import form_contours


manual_mask = None
manual_frame = None
manual_selected_contours = None
TEST = False


def safe_destroy_window(name):
    """cv2.destroyWindow raises if the window was never created (e.g. the automatic
    pass got the right count and manual segmentation never opened)."""
    try:
        cv2.destroyWindow(name)
    except cv2.error:
        pass


def get_contour_center(contour):
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def create_tracks(contours, prev_tracks):
    tracks = []
    assert contours, "No contours found"
    for contour in contours:
        center = get_contour_center(contour)
        if center is None:
            # Zero-area contour: no usable center, so it can't be matched by distance.
            continue
        tracks.append({"labels": [], "center": center, "contour": contour})
    return match_track_labels(tracks, prev_tracks)


def reactivate_retired_tracks(contours, previous_tracks, retired_labels, last_known_positions):
    """Peel off extra contours that best match a retired bee's last known position and
    give them back their original label, so a bee that leaves the retirement region
    later resumes being tracked under the same identity instead of being lost forever."""
    reactivated_tracks = []
    remaining_contours = list(contours)
    previous_count = len(previous_tracks or [])

    def distance_to(label, contour):
        center = get_contour_center(contour)
        if center is None:
            return float("inf")
        return cv2.norm(np.array(center) - np.array(last_known_positions[label]))

    while retired_labels and len(remaining_contours) > previous_count:
        best_label, best_index, best_dist = None, None, float("inf")
        for label in retired_labels:
            for index, contour in enumerate(remaining_contours):
                dist = distance_to(label, contour)
                if dist < best_dist:
                    best_label, best_index, best_dist = label, index, dist
        if best_label is None:  # every remaining candidate had a zero-area contour
            break

        # Pop by index: list.remove() compares NumPy arrays with ==, which raises a
        # broadcast error when contours have different point counts.
        closest_contour = remaining_contours.pop(best_index)
        retired_labels.discard(best_label)
        reactivated_tracks.append(
            {"labels": [best_label], "center": get_contour_center(closest_contour), "contour": closest_contour}
        )
    return reactivated_tracks, remaining_contours


def retire_tracks_at_region(tracks, region, retired_labels):
    region_x, region_y, region_width, region_height = region
    region_right = region_x + region_width
    region_bottom = region_y + region_height

    for track in tracks:
        if len(track["labels"]) != 1:
            continue
        x, y, width, height = cv2.boundingRect(track["contour"])
        touches_region = (
            x <= region_right
            and x + width >= region_x
            and y <= region_bottom
            and y + height >= region_y
        )
        if touches_region:
            retired_labels.add(track["labels"][0])

    active_tracks = []
    for track in tracks:
        track["labels"] = [
            label for label in track["labels"] if label not in retired_labels
        ]
        if track["labels"]:
            active_tracks.append(track)
    return active_tracks


def match_track_labels(tracks, prev_tracks):
    if prev_tracks is not None:
        if DEBUG:
            print("matching tracks", len(tracks), len(prev_tracks))
        if len(prev_tracks) > len(tracks):
            current_tracks = prev_tracks
            unmatched_tracks = tracks.copy()
        elif len(prev_tracks) == len(tracks):
            current_tracks = tracks
            unmatched_tracks = prev_tracks.copy()
        else:
            if DEBUG:
                print("Error: prev_tracks should never be less than tracks")
            return tracks

        for current_track in current_tracks:
            closest_track = sorted(
                unmatched_tracks,
                key=lambda track: cv2.norm(np.array(track["center"]) - np.array(current_track["center"])),
            )[0]
            if len(prev_tracks) > len(tracks):
                closest_track["labels"].extend(current_track["labels"])
            else:
                current_track["labels"].extend(closest_track["labels"])
                # Remove by identity: the tracks are dicts holding NumPy contours, so
                # list.remove() would compare them with == and can raise.
                unmatched_tracks = [t for t in unmatched_tracks if t is not closest_track]
        if DEBUG:
            print("test, all the labels in tracks", [track["labels"] for track in tracks])
    else:
        for index, track in enumerate(tracks):
            track["labels"].append(index)

    return [track for track in tracks if track["labels"]]


def draw_tracks(frame, tracks, show_label=True, show_area=False):
    for track in tracks:
        if len(track["labels"]) > 1 or not track["labels"]:
            color = (180, 180, 180)
        else:
            from utils.settings import COLORS

            color = COLORS[(track["labels"][0] + 1) % len(COLORS)]
        cv2.drawContours(frame, [track["contour"]], -1, color, 2)
        x, y, width, height = cv2.boundingRect(track["contour"])
        text = "Worker "
        if show_label:
            text += str([label + 1 for label in track["labels"]]).replace("[", "").replace("]", "")
        if show_area:
            text += ", Area " + str(cv2.contourArea(track["contour"]))
        cv2.putText(frame, text, (x + width, y + height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return frame


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame, processed, background_sub, points, manual_group_track, previous_contours, full_mask = param
        points.append((x, y))
        update_manual_frame(frame, processed, background_sub, points, manual_group_track, previous_contours, full_mask)


def update_manual_frame(frame, processed, background_sub, points, manual_group_track, previous_contours=None, full_mask=None):
    global manual_mask, manual_frame, manual_selected_contours

    # Detect contours across the whole frame (not just the isolated group region) up
    # front, so a click can claim a specific existing contour outright.
    contour_source = full_mask if full_mask is not None else processed
    contour_source_gray = cv2.cvtColor(contour_source, cv2.COLOR_BGR2GRAY)
    num_bees = len(manual_group_track["labels"]) if manual_group_track is not None else None
    current_contours, rejected_contours = form_contours(
        contour_source_gray,
        MIN_BEE_AREA,
        MAX_GROUP_AREA,
        NUM_BEES=num_bees,
        remove_background=False,
        remove_extra_contours=False,
        previous_contours=previous_contours,
        return_distance_rejected=True,
    )

    manual_mask = np.zeros(frame.shape, dtype=np.uint8)
    manual_selected_contours = []
    for point in points:
        # If the click lands inside an already-detected contour, claim that whole
        # contour as the selected bee instead of a color match, so each of the
        # num_bees points can land on a different, already-separated contour.
        selected_contour = next(
            (contour for contour in current_contours if cv2.pointPolygonTest(contour, point, False) >= 0),
            None,
        )
        manual_selected_contours.append(selected_contour)
        if selected_contour is not None:
            cv2.drawContours(manual_mask, [selected_contour], -1, (255, 255, 255), -1)
            continue
        bee_color = frame[point[1], point[0]]
        color_mask = cv2.cvtColor(color_threshold(frame, bee_color, MAX_THRESH_COLOR_DIFF), cv2.COLOR_GRAY2BGR)
        color_mask = cv2.bitwise_and(processed.astype(np.uint8), color_mask).astype(np.uint8)
        manual_mask = cv2.bitwise_or(manual_mask, color_mask)
    manual_frame = frame.copy()
    # Tint the detected foreground red without darkening the rest of the frame.
    red_tint = processed.copy()
    red_tint[:, :, 0] = 0
    red_tint[:, :, 1] = 0
    processed_pixels = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) > 0
    tinted_frame = cv2.addWeighted(frame, 0.8, red_tint, 0.2, 0)
    manual_frame[processed_pixels] = tinted_frame[processed_pixels]

    # Highlight the clicked selection on top, again without dimming the background.
    selection_pixels = cv2.cvtColor(manual_mask, cv2.COLOR_BGR2GRAY) > 0
    highlighted_frame = cv2.addWeighted(manual_frame, 0.7, manual_mask, 0.3, 0)
    manual_frame[selection_pixels] = highlighted_frame[selection_pixels]

    for point in points:
        cv2.circle(manual_frame, point, 5, (0, 0, 255), -1)
    if manual_group_track is not None:
        cv2.drawContours(manual_frame, [manual_group_track["contour"]], -1, (255, 255, 255), 2)

    cv2.drawContours(manual_frame, rejected_contours, -1, (255, 0, 255), 2)
    cv2.drawContours(manual_frame, current_contours, -1, (0, 255, 0), 2)
    for contour in current_contours:
        x, y, _, height = cv2.boundingRect(contour)
        area_label = f"Area: {cv2.contourArea(contour):.0f}"
        label_origin = (x, y - 6 if y > 20 else y + height + 16)
        cv2.putText(manual_frame, area_label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(manual_frame, area_label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)


def manual_segmentation(frame, processed, background_sub, manual_group_track, previous_contours=None, full_mask=None):
    global manual_mask, manual_frame, manual_selected_contours, ALLOW_MANUAL_SEGMENTING, MAX_THRESH_COLOR_DIFF, TEST
    points = []
    update_manual_frame(frame, processed, background_sub, points, manual_group_track, previous_contours, full_mask)
    while True:
        cv2.setMouseCallback(
            MANUAL_WINDOW_NAME,
            on_click,
            [frame, processed, background_sub, points, manual_group_track, previous_contours, full_mask],
        )
        cv2.imshow(MANUAL_WINDOW_NAME, manual_frame)
        key = cv2.waitKey(30) & 0xFF
        if key == 2:
            MAX_THRESH_COLOR_DIFF -= 1
            print("MAX_THRESH_COLOR_DIFF=", MAX_THRESH_COLOR_DIFF)
            update_manual_frame(frame, processed, background_sub, points, manual_group_track, previous_contours, full_mask)
        elif key == 3:
            MAX_THRESH_COLOR_DIFF += 1
            print("MAX_THRESH_COLOR_DIFF=", MAX_THRESH_COLOR_DIFF)
            update_manual_frame(frame, processed, background_sub, points, manual_group_track, previous_contours, full_mask)
        elif key == 0:
            points = []
            print("reset")
            break
        elif key == 13:
            print("enter")
            break
        elif key == ord("m"):
            ALLOW_MANUAL_SEGMENTING = not ALLOW_MANUAL_SEGMENTING
            print("manual segmentation toggled: ", ALLOW_MANUAL_SEGMENTING)
            break
        elif key == ord("d"):
            TEST = True
            print("dilate")
            break
        elif key == 26 and points:
            points.pop()
            update_manual_frame(frame, processed, background_sub, points, manual_group_track, previous_contours, full_mask)
            print("undo")
    close_auxiliary_windows()
    return points, manual_mask, manual_selected_contours


def _manual_group_mask(
    frame_num,
    settings,
    frame,
    background_sub,
    group_track,
    preprocess_data,
    previous_contours,
):
    settings = edit_preprocess_settings(
        frame_num,
        settings,
        frame,
        preprocess_data["src_processed_root"],
        keep_preview=True,
        previous_contours=previous_contours,
    )
    preprocess_data["settings_override"] = True
    refreshed_mask, background_sub = preprocess(frame, preprocess_data["frame_shape"], settings)

    # The new settings can shift or split the group's contour boundary, so re-detect it
    # by taking every new contour that overlaps the old region instead of a single
    # point-in-polygon match, which silently fell back to the stale contour whenever the
    # new settings moved the mask enough that the old center no longer fell inside it.
    old_region_mask = np.zeros(refreshed_mask.shape, dtype=np.uint8)
    cv2.drawContours(old_region_mask, [group_track["contour"]], -1, 255, -1)
    refreshed_contours, _ = cv2.findContours(refreshed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlapping_contours = []
    for contour in refreshed_contours:
        contour_mask = np.zeros(refreshed_mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        if cv2.countNonZero(cv2.bitwise_and(contour_mask, old_region_mask)):
            overlapping_contours.append(contour)

    region = np.zeros(refreshed_mask.shape, dtype=np.uint8)
    if overlapping_contours:
        cv2.drawContours(region, overlapping_contours, -1, 255, -1)
    else:
        region = old_region_mask

    updated_group_track = dict(group_track)
    region_contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(region_contours) == 1:
        updated_group_track["contour"] = region_contours[0]
        updated_group_track["center"] = get_contour_center(region_contours[0]) or group_track["center"]

    processed = cv2.bitwise_and(refreshed_mask, region)
    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    full_mask = cv2.cvtColor(refreshed_mask, cv2.COLOR_GRAY2BGR)
    update_manual_frame(frame, processed, background_sub, [], updated_group_track, previous_contours, full_mask)
    return settings, processed, background_sub, updated_group_track, full_mask


def split_groups(frame_num, settings, frame, background_sub, num_bees, groups, prev_tracks, preprocess_data):
    visualization = np.copy(frame)
    final_split_tracks = []
    if not preprocess_data.get("settings_override", False):
        settings = load_settings(preprocess_data["src_processed_root"], frame_num, default=settings)

    for group_track in groups:
        processed = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(processed, [group_track["contour"]], -1, 255, -1)
        previous_tracks = [
            previous_track
            for previous_track in prev_tracks
            if previous_track["labels"] and set(previous_track["labels"]).issubset(group_track["labels"])
        ]
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        point_colors = []
        points = []
        for previous_track in previous_tracks:
            point = previous_track["center"]
            points.append(point)
            point_colors.append(frame[point[1], point[0]])

        color_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for point_color in point_colors:
            color_mask = cv2.bitwise_or(color_mask, color_threshold(frame, point_color, MAX_THRESH_COLOR_DIFF))
        color_mask = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
        mask = cv2.bitwise_and(processed, color_mask).astype(np.uint8)

        correct_num_bees = False
        reset_group = False
        iteration = 0
        while not correct_num_bees:
            visualization = np.copy(frame)
            cv2.circle(visualization, group_track["center"], 5, (0, 0, 255), -1)
            markers = np.zeros(frame.shape[:2], dtype=np.int32)
            for index, point in enumerate(points):
                cv2.circle(markers, point, 1, index + 2, -1)
                cv2.circle(markers, (10, 10), 1, 1, -1)

            edges = cv2.watershed(mask, markers).astype(np.uint8)
            _, edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            bee_contours = form_contours(
                edges,
                MIN_BEE_AREA,
                MAX_BEE_AREA,
                num_bees,
                remove_background=True,
                remove_extra_contours=False,
                previous_contours=[track["contour"] for track in previous_tracks],
            )
            if len(bee_contours) == len(group_track["labels"]):
                correct_num_bees = True
                safe_destroy_window(MANUAL_WINDOW_NAME)

            cv2.drawContours(visualization, bee_contours, -1, (0, 255, 0), 2)
            for point in points:
                cv2.circle(visualization, point, 5, (0, 0, 255), -1)
            visualization = cv2.bitwise_or(
                cv2.multiply(mask, np.full(mask.shape[-1], 0.4)),
                cv2.multiply(visualization, np.full(mask.shape[-1], 0.7)),
            )

            if iteration == MAX_AUTOMATIC_ITERATIONS:
                if len(bee_contours) != len(group_track["labels"]) and ALLOW_MANUAL_SEGMENTING:
                    print("Error: Number of contours found is not ", len(group_track["labels"]), "number of contours found: ", len(bee_contours))
                    settings, processed, background_sub, group_track, full_mask = _manual_group_mask(
                        frame_num,
                        settings,
                        frame,
                        background_sub,
                        group_track,
                        preprocess_data,
                        [track["contour"] for track in prev_tracks],
                    )
                    points, mask, selected_contours = manual_segmentation(
                        frame,
                        processed,
                        background_sub,
                        group_track,
                        previous_contours=[track["contour"] for track in prev_tracks],
                        full_mask=full_mask,
                    )
                    if selected_contours and len(selected_contours) == len(group_track["labels"]) and all(
                        contour is not None for contour in selected_contours
                    ):
                        # Every click landed on its own already-distinct contour, so use them
                        # directly instead of running watershed, which collapses the separate
                        # regions back together once their small label values get binarized.
                        bee_contours = selected_contours
                        correct_num_bees = True
                        safe_destroy_window(MANUAL_WINDOW_NAME)
                    reset_group = not points and not correct_num_bees
                elif len(bee_contours) != len(group_track["labels"]):
                    reset_group = True
            else:
                # Rebuild the color mask with a threshold that shifts each pass. It was
                # previously left all-zero here, so every retry ran watershed on a black
                # image. Use "+ iteration" instead if the group is under-segmented and
                # you want to loosen rather than tighten.
                color_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                for point_color in point_colors:
                    color_mask = cv2.bitwise_or(
                        color_mask,
                        color_threshold(frame, point_color, max(0, MAX_THRESH_COLOR_DIFF - iteration)),
                    )
                color_mask = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
                mask = cv2.bitwise_and(processed, color_mask).astype(np.uint8)
                iteration += 1

            if reset_group:
                break

        if correct_num_bees and not reset_group:
            separated_tracks = []
            for bee_contour in bee_contours:
                separated_tracks.append({"labels": [], "center": get_contour_center(bee_contour), "contour": bee_contour})
            print("test new vs old track nums", len(separated_tracks), len(previous_tracks))
            final_split_tracks.extend(match_track_labels(separated_tracks, previous_tracks))

    return final_split_tracks, visualization, settings