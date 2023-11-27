import cv2
import numpy as np
import json
import utils.general as general_utils
import os
from settings import COLORS

VIDEO_NAME = ""
BACKGROUND_NAME = ""
ALLOW_MANUAL_LABELLING = False
ALLOW_MANUAL_SEGMENTING = True
MAX_AUTOMATIC_ITERATIONS = 15
WINDOW_NAME = "frame"
MANUAL_WINDOW_NAME = "manual point select - press ENTER to continue"
FRAMES_PATH = "denoised_frames"
MIN_BEE_AREA = 80
MAX_BEE_AREA = 1500
MIN_GROUP_AREA = 100
MAX_GROUP_AREA = 5000
MAX_MOVE_DISTANCE = 100
MAX_THRESH_COLOR_DIFF = 60

NUM_BEES = 2

def imgs2vid(imgs, outpath, fps):
    ''' Stitch together frame imgs to make a movie. '''
    height, width, layers = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(outpath, fourcc, fps, (width, height), True)

    for img_i, img in enumerate(imgs):
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

def get_contour_center(contour):
    bee_moment = cv2.moments(contour)
    if bee_moment["m00"] == 0:
        return None
    cX = int(bee_moment["m10"] / bee_moment["m00"])
    cY = int(bee_moment["m01"] / bee_moment["m00"])
    center = (cX, cY)
    return center

def preprocess(frame, background):
    n = float(3.0)
    img = cv2.multiply(frame, np.array([n]))
    img_background = cv2.multiply(background, np.array([n]))

    # Background Subtraction
    background_sub = cv2.cvtColor(cv2.absdiff(img, img_background), cv2.COLOR_BGR2GRAY)
    cv2.imshow("bg", background_sub)
    blurred = cv2.GaussianBlur(background_sub, (25, 25), 0) # blur to quickly denoise, precision is not important for this step
    _, threshold = cv2.threshold(blurred,thresh=35,maxval=255,type=cv2.THRESH_BINARY)
    erode = cv2.erode(threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    dilate = cv2.dilate(erode, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=8)
    cv2.rectangle(dilate, (500, 30, 75, 370), 0, -1)
    cv2.rectangle(dilate, (1245, 105, 20, 20), 0, -1) # random artifact
    cv2.imshow("thresh", dilate)
    return dilate, background_sub

def form_contours(image, minArea, maxArea, remove_background=False, remove_extra_contours=False):
    global min_contour
    # Form contours and filter by size
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # remove background and empty space
    if len(contours) == 0 or len(hierarchy) == 0:
        return contours
    hierarchy = hierarchy[0]

    # Remove background and neutral space
    if remove_background:
        contours = [contour for i, contour in enumerate(contours) if hierarchy[i][2] == -1]

    prev_num_contours = len(contours)
    contours = [contour for contour in contours if minArea < cv2.contourArea(contour) < maxArea]
    if not prev_num_contours == len(contours):
        print("Warning: Removed contours based on area")

    if remove_extra_contours:
        if len(contours) > NUM_BEES:
            diff = len(contours) - NUM_BEES
            contours = sorted(contours, key=lambda contour: cv2.contourArea(contour))[diff:]
            print("Warning: Removed excess contours")
    return contours

def match_track_labels(tracks, prev_tracks):
    if prev_tracks is not None: # if a group has been formed, i.e. no longer 1:1 matching
        print("matching tracks", len(tracks), len(prev_tracks))
        current_tracks = None
        unmatched_tracks = None
        # NOTE: ideally there would always be 1:1 matching, but sometimes a group is split or a group is formed even with the watershed algorithm
        if len(prev_tracks) > len(tracks): # if a group was formed, just match to whatever's closest, i.e. 1:n matching
            current_tracks =  prev_tracks
            unmatched_tracks = tracks.copy()
        elif len(prev_tracks) == len(tracks): # if a group was split or nothing changed, match to whatever's closest and not already matched, i.e. n:1 or 1:1 matching
            current_tracks = tracks
            unmatched_tracks = prev_tracks.copy()
        else:
            # NOTE: if a group was split that is the worst case scenario, because we cannot tell which bees belong to which group
            # TODO: add manual matching
            print("Error: prev_tracks should never be less than tracks")
            return tracks

        for current_track in current_tracks:
            closest_track = sorted(unmatched_tracks, key=lambda track: cv2.norm(np.array(track["center"]) - np.array(current_track["center"])))[0]
            if len(prev_tracks) > len(tracks):
                closest_track["labels"].extend(current_track["labels"])
            if len(prev_tracks) == len(tracks): # if a group wasn't formed, i.e. n:1 or 1:1 matching
                current_track["labels"].extend(closest_track["labels"])
                unmatched_tracks.remove(closest_track)
        print("test", [final_track["labels"] for final_track in tracks])

    else: # TODO, assign based on min move distance
        for i, track in enumerate(tracks):
            track["labels"].append(i)

    final_tracks = []
    for track in tracks:
        if not len(track["labels"]) == 0:
            final_tracks.append(track)

    return final_tracks

def create_tracks(contours, prev_tracks):
    tracks = []
    assert not len(contours) == 0
    for i, contour in enumerate(contours):
        track_center = get_contour_center(contour)
        track = {"labels": [], "center": track_center, "contour": contour}
        tracks.append(track)

    # Assign Labels
    tracks = match_track_labels(tracks, prev_tracks)
    return tracks

def draw_tracks(frame, tracks, show_label=True, show_area=False):
    # Draw contours
    for track in tracks:
        if len(track["labels"]) > 1 or len(track["labels"]) == 0:
            color = (180, 180, 180)
        else:
            color = COLORS[(track["labels"][0]+1) % len(COLORS)]
        cv2.drawContours(frame, [track["contour"]], -1, color, 2)

        x,y,w,h = cv2.boundingRect(track["contour"])
        bottom_right = (x+w, y+h)
        track_text = "Worker "
        if show_label:
            track_text = track_text + str([label+1 for label in track["labels"]]).replace("[", "").replace("]", "") # 1 indexed to make it easier to read
        if show_area:
            track_text = track_text + ", Area " + str(cv2.contourArea(track["contour"]))
        frame = cv2.putText(frame, track_text, bottom_right, cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2, cv2.LINE_AA)
    return frame

def color_threshold(image, color, offfset):
    color_thresh = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    color_thresh = cv2.bitwise_or(color_thresh, cv2.inRange(image, int(color)-offfset,  255))
    return color_thresh

def on_click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param[0]
        processed = param[1]
        background_sub = param[2]
        points = param[3]
        manual_group_track = param[4]
        points.append((x,y))
        update_manual_frame(frame, processed, background_sub, points, manual_group_track)

def update_manual_frame(frame, processed, background_sub, points, manual_group_track):
    global manual_mask
    global manual_frame
    manual_mask = np.zeros(frame.shape, dtype=np.uint8)
    for point in points:
        bee_color = background_sub[point[1], point[0]]
        color_thresh = cv2.cvtColor(color_threshold(background_sub, bee_color, MAX_THRESH_COLOR_DIFF), cv2.COLOR_GRAY2BGR)
        color_thresh = cv2.bitwise_and(processed, color_thresh).astype(np.uint8)
        manual_mask = cv2.bitwise_or(manual_mask, color_thresh)
    manual_frame = cv2.bitwise_or(cv2.multiply(manual_mask, np.array([.3])), cv2.multiply(frame, np.array([.7])))
    for point in points:
        manual_frame = cv2.circle(manual_frame, point, 5, (0, 0, 255), -1)
    red_mask = processed.copy()
    red_mask[:,:,0] = 0
    red_mask[:,:,1] = 0
    manual_frame = cv2.bitwise_or(cv2.multiply(red_mask, np.array([.2])), cv2.multiply(manual_frame, np.array([.8])))
    manual_frame = cv2.drawContours(manual_frame, [manual_group_track["contour"]], -1, (255, 255, 255), 2)

manual_mask = None
manual_frame = None
def manual_segmentation(frame, processed, background_sub, manual_group_track):
    global MAX_THRESH_COLOR_DIFF
    global manual_mask
    global manual_frame
    global ALLOW_MANUAL_SEGMENTING
    global TEST
    points = []

    update_manual_frame(frame, processed, background_sub, points, manual_group_track)
    while(1):
        # mask = np.copy(dilate)
        cv2.setMouseCallback(MANUAL_WINDOW_NAME, on_click, [frame, processed, background_sub, points, manual_group_track])
        cv2.imshow(MANUAL_WINDOW_NAME, manual_frame)
        k = cv2.waitKey(30) & 0xFF

        if k == 2: # left arrow
            MAX_THRESH_COLOR_DIFF = MAX_THRESH_COLOR_DIFF - 1
            print("MAX_THRESH_COLOR_DIFF=",MAX_THRESH_COLOR_DIFF)
        elif k == 3: # right arrow
            MAX_THRESH_COLOR_DIFF = MAX_THRESH_COLOR_DIFF + 1
            print("MAX_THRESH_COLOR_DIFF=",MAX_THRESH_COLOR_DIFF)
        elif k == 0: # down arrow
            points = [] # breaking with no points will trigger a reset
            print("reset")
            break
        elif k == 13: # enter
            print("enter")
            break
        elif k == 109: # m key
            ALLOW_MANUAL_SEGMENTING = not ALLOW_MANUAL_SEGMENTING
            print("manual segmentation toggled: ", ALLOW_MANUAL_SEGMENTING)
            break
        # d key
        elif k == 100:
            TEST = True
            print("dilate")
            break
        elif k == 26: # Control + Z
            if not len(points) == 0:
                points.pop()
                update_manual_frame(frame, processed, background_sub, points, manual_group_track)
                print("undo")

    return points, manual_mask

def split_groups(frame, background_sub, groups, prev_tracks):
    visualization = np.copy(frame)
    final_split_tracks = []
    # Split groups
    for group_track in groups:
        processed = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        processed = cv2.drawContours(processed, [group_track["contour"]], -1, (255, 255, 255), -1)

        # Get the individual tracks that are apart of the group
        # NOTE: This is needed so we can isolate the bees that are apart of the group
        prev_unmerged_tracks = []
        for prev_track in prev_tracks:
            if not len(prev_track["labels"]) == 0 and set(prev_track["labels"]).issubset(group_track["labels"]):
                prev_unmerged_tracks.append(prev_track)

        if not len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        # Get the colors of the bees in the group for color thresholding
        point_colors = []
        points = []
        for prev_unmerged_track in prev_unmerged_tracks:
            point = prev_unmerged_track["center"]
            points.append(point)
            point_colors.append(background_sub[point[1], point[0]])

        # Threshold based on the colors of the points
        color_thresh = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        for point_color in point_colors:
            color_thresh = cv2.bitwise_or(color_thresh, color_threshold(background_sub, point_color, MAX_THRESH_COLOR_DIFF))
        color_thresh = cv2.cvtColor(color_thresh, cv2.COLOR_GRAY2BGR)
        mask = np.copy(cv2.bitwise_and(processed, color_thresh).astype(np.uint8))

        # Apply watershed algorithm
        correct_num_bees = False
        reset_group = False
        iteration = 0
        # Either iteratively color threshold or manually segment until the correct number of bees are found
        while not correct_num_bees:
            visualization = np.copy(frame)
            cv2.circle(visualization, group_track["center"], 5, (0, 0, 255), -1)

            markers = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.int32)
            for i, point in enumerate(points):
                # Add two to the color to make room for the background and neutral space
                markers = cv2.circle(markers, point, 1, i+2, -1)
                markers = cv2.circle(markers, (10,10), 1, 1, -1)

            edges = cv2.watershed(mask, markers).astype(np.uint8)
            _, edges = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU) # needed to find contours
            # cv2.imshow('frame4', (edges*50).astype(np.uint8))
            bee_contours = form_contours(edges, MIN_BEE_AREA, MAX_BEE_AREA, remove_background=True, remove_extra_contours=False)

            cv2.drawContours(visualization, bee_contours, -1, (0, 255, 0), 2)
            for point in points:
                cv2.circle(visualization, point, 5, (0, 0, 255), -1)
            visualization = cv2.bitwise_or(cv2.multiply(mask, np.array([.4])), cv2.multiply(visualization, np.array([.7])))

            # Iteratively color threshold until the correct number of bees are found or the max number of iterations is reached
            if iteration == MAX_AUTOMATIC_ITERATIONS:
                if not len(bee_contours) == len(group_track["labels"]) and ALLOW_MANUAL_SEGMENTING:
                    print("Error: Number of contours found is not ", len(group_track["labels"]), "number of contours found: ", len(bee_contours))
                    # global_group_track = group_track
                    custom_points, custom_mask = manual_segmentation(frame, processed, background_sub, group_track)
                    reset_group = len(custom_points) == 0
                    points = custom_points
                    mask = custom_mask
                elif not len(bee_contours) == len(group_track["labels"]) and not ALLOW_MANUAL_SEGMENTING:
                    reset_group = True
            else:
                color_thresh = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                for point_color in point_colors:
                    color_thresh = cv2.bitwise_or(color_thresh, color_threshold(background_sub, point_color, MAX_THRESH_COLOR_DIFF - iteration))
                color_thresh = cv2.cvtColor(color_thresh, cv2.COLOR_GRAY2BGR)
                mask = np.copy(cv2.bitwise_and(processed, color_thresh).astype(np.uint8))
                iteration += 1

            if len(bee_contours) == len(group_track["labels"]):
                correct_num_bees = True
                cv2.destroyWindow(MANUAL_WINDOW_NAME)

            if reset_group:
                break

        if correct_num_bees and not reset_group:
            separated_tracks = []
            for bee_contour in bee_contours:
                bee_center = get_contour_center(bee_contour)
                separated_track = {"labels": [], "center": bee_center, "contour": bee_contour}
                separated_tracks.append(separated_track)

            # NOTE: this should always be a 1:1 matching if watershed was successful
            print("test", len(separated_tracks), len(prev_unmerged_tracks))
            labelled_separated_tracks = match_track_labels(separated_tracks, prev_unmerged_tracks)
            final_split_tracks.extend(labelled_separated_tracks)
    return final_split_tracks, visualization

if __name__ == "__main__":
    print("select root folder...")
    src_processed_root = general_utils.select_file("data/processed")
    print("select video from list...")
    video = general_utils.select_file(src_processed_root)
    print("select background file from list...")
    background = general_utils.select_file(src_processed_root)

    VIDEO_NAME = video.split("/")[-1].replace(".mp4", "")
    BACKGROUND_NAME = background.split("/")[-1].replace(".png", "")

    cap = cv2.VideoCapture(video)
    background = cv2.imread(background)

    data_log = {}
    raw_frames = []
    annotated_frames = []
    prev_tracks = None
    while (cap.isOpened()):
        success, frame = cap.read()

        if not success:
            break
        raw_frames.append(frame)
        processed, background_sub = preprocess(frame, background)

        # Pass in MAX_GROUP_AREA in case groups were formed
        contours = form_contours(processed, MIN_BEE_AREA, MAX_GROUP_AREA, remove_background=False, remove_extra_contours=True)

        if len(contours) == 0:
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

        # Split the groups and update the tracks
        if ALLOW_MANUAL_SEGMENTING and (prev_tracks is not None) and len(groups) >= 1:
            # frame, split_tracks = split_groups(frame, background_sub, tracks, prev_tracks)
            # TODO: test new structure (uncomment above and comment below for new structure)
            split_tracks, frame = split_groups(frame, background_sub, groups, prev_tracks)
            for group in groups:
                tracks.remove(group)
            tracks.extend(split_tracks)

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
        if k == 32:
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
    imgs2vid(annotated_frames, video_outpath, 30)
    if not os.path.exists(f'{src_processed_root}/{FRAMES_PATH}'):
        os.mkdir(f'{src_processed_root}/{FRAMES_PATH}')
        for i, raw_frame in enumerate(raw_frames):
            cv2.imwrite(f'{src_processed_root}/{FRAMES_PATH}/frame_{i+1:05d}.png', raw_frame)
    cap.release()
    cv2.destroyAllWindows()
