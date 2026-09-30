import json
import os

import cv2
import numpy as np

import utils.general as general_utils
from utils.preprocess import (
    close_auxiliary_windows,
    default_preprocess_settings,
    edit_preprocess_settings,
    load_settings,
    preprocess,
    set_preprocess_settings,
)
from utils.settings import (
    ALLOW_MANUAL_SEGMENTING,
    COLORS,
    FRAMES_PATH,
    LOAD_EXISTING_PREPROCESS_SETTINGS,
    MAX_BEE_AREA,
    MAX_GROUP_AREA,
    MIN_BEE_AREA,
    BEE_RETIREMENT_REGION,
    WINDOW_NAME,
)
from utils.tracker import (
    create_tracks,
    draw_tracks,
    manual_segmentation,
    reactivate_retired_tracks,
    retire_tracks_at_region,
    split_groups,
)


def record_positions(bee_positions, tracks):
    for track in tracks:
        if len(track["labels"]) != 1:
            continue
        label = track["labels"][0]
        bee_positions.setdefault(label, {"positions": [], "color": COLORS[(label + 1) % len(COLORS)]})
        bee_positions[label]["positions"].append(track["center"])


def build_tracks(contours, previous_tracks, retired_labels, bee_positions):
    # A retired bee that leaves the retirement region again shows up as an extra
    # contour beyond the currently active tracks; reclaim it under its old label
    # before handing the rest off to the normal nearest-previous-track matching.
    last_known_positions = {label: data["positions"][-1] for label, data in bee_positions.items()}
    reactivated_tracks, remaining_contours = reactivate_retired_tracks(
        contours, previous_tracks, retired_labels, last_known_positions
    )
    tracks = create_tracks(remaining_contours, previous_tracks) if remaining_contours else []
    tracks.extend(reactivated_tracks)
    record_positions(bee_positions, tracks)
    tracks = retire_tracks_at_region(tracks, BEE_RETIREMENT_REGION, retired_labels)
    return tracks


def draw_trails(frame, bee_positions):
    for position_data in bee_positions.values():
        positions = position_data["positions"]
        color = position_data["color"]
        for i in range(1, len(positions)):
            cv2.line(frame, positions[i], positions[i - 1], color, 2, cv2.LINE_AA)
    return frame


def _position_reference_contour(position, size=3):
    x, y = position
    return np.array(
        [[[x - size, y - size]], [[x + size, y - size]], [[x + size, y + size]], [[x - size, y + size]]],
        dtype=np.int32,
    )


def build_reference_contours(previous_tracks, retired_labels, bee_positions):
    # A bee that reappears near the retirement region is often far from the other
    # still-active bee, so also include retired bees' last known spots here or the
    # distance-based contour filter would discard the reappearance before it's ever seen.
    reference_contours = [track["contour"] for track in previous_tracks or []]
    for label in retired_labels:
        if label in bee_positions:
            reference_contours.append(_position_reference_contour(bee_positions[label]["positions"][-1]))
    return reference_contours


if __name__ == "__main__":
    # Select the source video and open it for frame-by-frame processing.
    print("select root folder...")
    src_processed_root = general_utils.select_file("data/processed")
    print("select processed video from list...")
    video = general_utils.select_file(src_processed_root)

    video_name = os.path.basename(video).replace(".mp4", "")
    cap = cv2.VideoCapture(video)

    # Collect tracking results, original frames, and rendered preview frames.
    data_log = {}
    raw_frames = []
    annotated_frames = []
    previous_tracks = None

    # Read the initial frame to establish dimensions and initialize settings.
    success, frame = cap.read()
    frame_shape = (frame.shape[1], frame.shape[0])
    frame_num = 0

    # Reuse saved preprocessing values, or configure and save them on first use.
    settings = load_settings(src_processed_root, frame_num)
    if not LOAD_EXISTING_PREPROCESS_SETTINGS:
        settings = default_preprocess_settings(frame_num)
        cv2.imshow("frame", frame)
        points, _, _ = manual_segmentation(
            frame,
            np.zeros(frame.shape, dtype=np.uint8),
            np.full((frame.shape[0], frame.shape[1], 1), 255, dtype=np.uint8),
            None,
        )
        settings["num_bees"] = len(points)
        settings["bee_colors"] = [frame[y, x].tolist() for x, y in points]
        set_preprocess_settings(frame_num, settings, cap, frame, frame_shape, src_processed_root)
        settings = load_settings(src_processed_root, frame_num, default=settings)
    # print("Preprocess settings loaded for frame", frame_num, "are", settings)
    # exit()
    preprocess_settings_overridden = False
    retired_labels = set()
    bee_positions = {}

    # Process each video frame: preprocess it, detect bee contours, and track them.
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        raw_frames.append(frame)
        # Detect at the full expected count (not reduced by retirements) so a bee
        # that leaves the retirement region again isn't truncated away as "extra".
        processed, background_sub = preprocess(frame, frame_shape, settings)
        reference_contours = build_reference_contours(previous_tracks, retired_labels, bee_positions)
        contours = general_utils.form_contours(
            processed,
            MIN_BEE_AREA,
            MAX_GROUP_AREA,
            NUM_BEES=settings["num_bees"],
            remove_background=False,
            remove_extra_contours=True,
            previous_contours=reference_contours,
        )

        # Let the operator tune preprocessing if the current settings find nothing,
        # unless some bees have already retired, in which case an empty frame just
        # means nobody currently active is visible right now (not a settings problem).
        if not contours and not retired_labels:
            print("Warning: No contours were created")
            settings = edit_preprocess_settings(
                frame_num,
                settings,
                frame,
                src_processed_root,
                keep_preview=True,
                previous_contours=reference_contours,
            )

            preprocess_settings_overridden = True
            processed, background_sub = preprocess(frame, frame_shape, settings)
            contours = general_utils.form_contours(
                processed,
                MIN_BEE_AREA,
                MAX_GROUP_AREA,
                NUM_BEES=settings["num_bees"],
                remove_background=False,
                remove_extra_contours=True,
                previous_contours=reference_contours,
            )
            # Tuning is done; close the preview windows so they don't linger over
            # subsequent frames once normal tracking resumes.
            close_auxiliary_windows()

        # If tuning still finds no contours, request manual bee-color selections.
        # if not contours:
        #     selection_mask = np.full(frame.shape, 255, dtype=np.uint8)
        #     while not contours:
        #         print("Manually select bee colors, then press Enter. Repeat until contours are detected.")
        #         points, selected_mask = manual_segmentation(frame, selection_mask, background_sub, None)
        #         if not points:
        #             print("No points selected; manual segmentation is required to continue.")
        #             continue
        #         selected_binary = cv2.cvtColor(selected_mask, cv2.COLOR_BGR2GRAY)
        #         contours = general_utils.form_contours(
        #             selected_binary,
        #             MIN_BEE_AREA,
        #             MAX_GROUP_AREA,
        #             NUM_BEES=settings["num_bees"],
        #             remove_background=False,
        #             remove_extra_contours=False,
        #         )
        #         if not contours:
        #             print("Manual selection produced no usable contours; please try again.")

        if not contours:
            # Nothing detected and nobody's expected to be visible; nothing to track.
            annotated_frames.append(frame)
            frame_key = f"frame_{len(annotated_frames):05d}"
            data_log[frame_key] = []
            previous_tracks = []
            frame_num += 1
            continue

        # Match detections to persistent bee labels; draw_tracks renders their outlines.
        tracks = build_tracks(contours, previous_tracks, retired_labels, bee_positions)
        active_num_bees = max(0, settings["num_bees"] - len(retired_labels))
        if len(tracks) == 0:
            print("Warning: No tracks were created")
            if active_num_bees == 0:
                previous_tracks = []
            annotated_frames.append(frame)
            frame_key = f"frame_{len(annotated_frames):05d}"
            data_log[frame_key] = []
            frame_num += 1
            continue

        # A track with multiple labels represents bees currently merged into a group.
        groups = [track for track in tracks if len(track["labels"]) > 1]

        # A label can also get folded into another track's group simply because that
        # bee wasn't detected this frame (e.g. it's in a corner/artifact region), not
        # because it's actually merged with another bee. Treat that as a missed
        # detection to fix via settings, not as a group to watershed-split.
        fake_groups = [group for group in groups if cv2.contourArea(group["contour"]) <= MAX_BEE_AREA]
        if fake_groups:
            print("Warning: a bee appears undetected rather than merged; adjust preprocessing settings")
            reference_contours = build_reference_contours(previous_tracks, retired_labels, bee_positions)
            settings = edit_preprocess_settings(
                frame_num,
                settings,
                frame,
                src_processed_root,
                keep_preview=True,
                previous_contours=reference_contours,
            )
            preprocess_settings_overridden = True
            processed, background_sub = preprocess(frame, frame_shape, settings)
            contours = general_utils.form_contours(
                processed,
                MIN_BEE_AREA,
                MAX_GROUP_AREA,
                NUM_BEES=settings["num_bees"],
                remove_background=False,
                remove_extra_contours=True,
                previous_contours=reference_contours,
            )
            close_auxiliary_windows()
            tracks = build_tracks(contours, previous_tracks, retired_labels, bee_positions)
            active_num_bees = max(0, settings["num_bees"] - len(retired_labels))
            groups = [track for track in tracks if len(track["labels"]) > 1]

        if groups:
            print("group")
            for group in groups:
                print(group["labels"])
            print("")
        print("length", len(groups))

        # Optionally split merged groups using prior tracks and manual guidance.
        if ALLOW_MANUAL_SEGMENTING and previous_tracks is not None and groups:
            preprocess_data = {
                "cap": cap,
                "frame": frame,
                "frame_shape": frame_shape,
                "src_processed_root": src_processed_root,
                "settings_override": preprocess_settings_overridden,
            }
            split_tracks, frame, settings = split_groups(
                frame_num,
                settings,
                frame,
                background_sub,
                active_num_bees,
                groups,
                previous_tracks,
                preprocess_data,
            )
            preprocess_settings_overridden = preprocess_data["settings_override"]
            for group in groups:
                tracks.remove(group)
            tracks.extend(split_tracks)
            record_positions(bee_positions, split_tracks)

        # Render labels and contours in the live preview and keep the rendered frame.
        frame = draw_trails(frame, bee_positions)
        frame = draw_tracks(frame, tracks)
        cv2.imshow(WINDOW_NAME, frame)

        annotated_frames.append(frame)
        frame_key = f"frame_{len(annotated_frames):05d}"
        data_log[frame_key] = []
        # Store each track's labels and bounding box for downstream pipeline stages.
        for track in tracks:
            x, y, width, height = cv2.boundingRect(track["contour"])
            is_group = track in groups
            data_log[frame_key].append(
                {
                    "label": str(track["labels"]).replace("[", "").replace("]", ""),
                    "x": x,
                    "y": y,
                    "h": height,
                    "w": width,
                    "id": "cluster" if is_group else "individual",
                }
            )

        # Use the current tracks as the reference for the next frame, then handle UI keys.
        previous_tracks = tracks
        key = cv2.waitKey(30) & 0xFF
        if key == 32:
            cv2.waitKey(0)
        if key == ord("m"):
            ALLOW_MANUAL_SEGMENTING = not ALLOW_MANUAL_SEGMENTING
            print("manual segmentation toggled: ", ALLOW_MANUAL_SEGMENTING)
            break
        if key == 27:
            break
        frame_num += 1

    # Export tracking data and the annotated video; save raw frames if not already present.
    print("Exporting data log...")
    datalog_outpath = os.path.join(src_processed_root, "data_log.json")
    with open(datalog_outpath, "w", encoding="utf-8") as outfile:
        json.dump(data_log, outfile)
    print("Exporting video...")
    video_outpath = os.path.join(src_processed_root, f"{video_name}_contours.mp4")
    general_utils.imgs2vid(annotated_frames, video_outpath, 25)
    frames_directory = os.path.join(src_processed_root, FRAMES_PATH)
    if not os.path.exists(frames_directory):
        os.mkdir(frames_directory)
        for index, raw_frame in enumerate(raw_frames):
            cv2.imwrite(os.path.join(frames_directory, f"frame_{index + 1:05d}.png"), raw_frame)
    cap.release()
    cv2.destroyAllWindows()
