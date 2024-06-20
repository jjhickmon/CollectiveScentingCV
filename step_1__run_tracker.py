import cv2
import numpy as np
import json
import utils.general as general_utils
import os
from utils.settings import *
from utils.preprocess import *
from utils.tracker import *
from utils.image import *
from utils.ffmpeg import *
from utils.general import *


if __name__ == "__main__":
    print("select root folder...")
    src_processed_root = general_utils.select_file("data/processed")
    print("select processed video from list...")
    video = general_utils.select_file(src_processed_root)

    VIDEO_NAME = video.split("/")[-1].replace(".mp4", "")

    cap = cv2.VideoCapture(video)

    data_log = {}
    raw_frames = []
    annotated_frames = []
    prev_tracks = None

    success, frame = cap.read()
    frame_shape = (frame.shape[1], frame.shape[0])

    if not LOAD_PREPROCESS_SETTINGS:
        set_preprocess_settings(cap, frame, frame_shape, src_processed_root)
    with open(f"{src_processed_root}/preprocess_settings.json") as json_file:
        settings = json.load(json_file)
        color_multiplier = settings["color_multiplier"]
        color_thresh = settings["color_thresh"]
        bee_colors = settings["bee_colors"]
        thresh = settings["thresh"]
        dilate_iter = settings["dilate_iter"]
        erode_iter = settings["erode_iter"]
        artifacts = settings["artifacts"]
        NUM_BEES = settings["num_bees"]
        print("Loaded settings: ", settings)

    while (cap.isOpened()):
        success, frame = cap.read()

        if not success:
            break
        raw_frames.append(frame)
        processed, background_sub = preprocess(frame, frame_shape, color_multiplier, thresh, color_thresh, bee_colors, dilate_iter, erode_iter, artifacts)

        # Pass in MAX_GROUP_AREA in case groups were formed
        contours = form_contours(processed, MIN_BEE_AREA, MAX_GROUP_AREA, NUM_BEES=NUM_BEES, remove_background=False, remove_extra_contours=True)

        # draw contours using open cv
        frame_contours = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        if len(frame_contours) == 0:
            print("Warning: No contours were created")
            continue
        tracks = create_tracks(contours, prev_tracks)
        if len(tracks) == 0:
            print("Warning: No tracks were created")
            continue

        # Create groups
        groups = [track for track in tracks if len(track["labels"]) > 1]
        if len(groups) >= 1:
            print("group")
            for group in groups:
                print(group["labels"])
            print("")
        print("length", len(groups))

        # cv2.imshow("test", frame_contours)
        # cv2.waitKey(0)

        # Split the groups and update the tracks
        if ALLOW_MANUAL_SEGMENTING and (prev_tracks is not None) and len(groups) >= 1:
            split_tracks, frame = split_groups(frame, background_sub, NUM_BEES, groups, prev_tracks)
            for group in groups:
                tracks.remove(group)
            tracks.extend(split_tracks)
        # else:
        #     tracks = match_track_labels(tracks, prev_tracks)

        frame = draw_tracks(frame, tracks, show_area=False)
        cv2.imshow(WINDOW_NAME, frame)

        annotated_frames.append(frame)
        data_log[f"frame_{len(annotated_frames):05d}"] = []
        for track in tracks:
            x,y,w,h = cv2.boundingRect(track["contour"])
            isgroup = track in groups
            data = { "label": str(track["labels"]).replace('[','').replace(']',''), "x": x, "y": y, "h": h, "w": w, "id": "individual" if not isgroup else "cluster" }
            data_log[f"frame_{len(annotated_frames):05d}"].append(data)

        prev_tracks = tracks

        k = cv2.waitKey(30) & 0xff
        if k == 32: # space key to pause
            cv2.waitKey(0)
        if k == 109: # m key
            ALLOW_MANUAL_SEGMENTING = not ALLOW_MANUAL_SEGMENTING
            print("manual segmentation toggled: ", ALLOW_MANUAL_SEGMENTING)
            break
        if k == 27:
            break

    print("Exporting data log...")
    datalog_outpath = f"{src_processed_root}/data_log.json"
    with open(datalog_outpath, "w") as outfile:
        json.dump(data_log, outfile)
    print("Exporting video...")
    video_outpath = f"{src_processed_root}/{VIDEO_NAME}_contours.mp4"
    imgs2vid(annotated_frames, video_outpath, 25)
    if not os.path.exists(f'{src_processed_root}/{FRAMES_PATH}'):
        os.mkdir(f'{src_processed_root}/{FRAMES_PATH}')
        for i, raw_frame in enumerate(raw_frames):
            cv2.imwrite(f'{src_processed_root}/{FRAMES_PATH}/frame_{i+1:05d}.png', raw_frame)
    cap.release()
    cv2.destroyAllWindows()
