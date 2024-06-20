import cv2
from utils.settings import *
import numpy as np
from utils.image import color_threshold
from utils.settings import *
from utils.general import *

def get_contour_center(contour):
    bee_moment = cv2.moments(contour)
    if bee_moment["m00"] == 0:
        return None
    cX = int(bee_moment["m10"] / bee_moment["m00"])
    cY = int(bee_moment["m01"] / bee_moment["m00"])
    center = (cX, cY)
    return center

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

# TODO: this will need to be improved as stated in future work
def match_track_labels(tracks, prev_tracks):
    if prev_tracks is not None: # if a group has been formed, i.e. no longer 1:1 matching
        if DEBUG:
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
            if DEBUG:
                print("Error: prev_tracks should never be less than tracks")
            return tracks

        for current_track in current_tracks:
            closest_track = sorted(unmatched_tracks, key=lambda track: cv2.norm(np.array(track["center"]) - np.array(current_track["center"])))[0]
            if len(prev_tracks) > len(tracks):
                closest_track["labels"].extend(current_track["labels"])
            if len(prev_tracks) == len(tracks): # if a group wasn't formed, i.e. n:1 or 1:1 matching
                current_track["labels"].extend(closest_track["labels"])
                unmatched_tracks.remove(closest_track)
        if DEBUG:
            print("test", [final_track["labels"] for final_track in tracks])

    else: # TODO, assign based on min move distance
        for i, track in enumerate(tracks):
            track["labels"].append(i)

    final_tracks = []
    for track in tracks:
        if not len(track["labels"]) == 0:
            final_tracks.append(track)

    return final_tracks

def draw_tracks(frame, tracks, show_label=True, show_area=False):
    # Draw contours
    for track in tracks:
        if len(track["labels"]) > 1 or len(track["labels"]) == 0:
            color = (180, 180, 180)
        else:
            color = COLORS[(track["labels"][0]+1) % len(COLORS)]

        frame = cv2.drawContours(frame, [track["contour"]], -1, color, 2)

        x,y,w,h = cv2.boundingRect(track["contour"])
        bottom_right = (x+w, y+h)
        track_text = "Worker "
        if show_label:
            track_text = track_text + str([label+1 for label in track["labels"]]).replace("[", "").replace("]", "") # 1 indexed to make it easier to read
        if show_area:
            track_text = track_text + ", Area " + str(cv2.contourArea(track["contour"]))
        frame = cv2.putText(frame, track_text, bottom_right, cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2, cv2.LINE_AA)
    return frame

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
        bee_color = frame[point[1], point[0]]
        color_thresh = cv2.cvtColor(color_threshold(frame, bee_color, MAX_THRESH_COLOR_DIFF), cv2.COLOR_GRAY2BGR)
        color_thresh = cv2.bitwise_and(processed.astype(np.uint8), color_thresh).astype(np.uint8)
        manual_mask = cv2.bitwise_or(manual_mask, color_thresh)
    manual_frame = cv2.bitwise_or(cv2.multiply(manual_mask, np.full(manual_mask.shape[-1], .3)), cv2.multiply(frame, np.full(frame.shape[-1], .7)))
    for point in points:
        manual_frame = cv2.circle(manual_frame, point, 5, (0, 0, 255), -1)
    red_mask = processed.copy()
    red_mask[:,:,0] = 0
    red_mask[:,:,1] = 0
    manual_frame = cv2.bitwise_or(cv2.multiply(red_mask, np.full(red_mask.shape[-1], .2)).astype(np.uint8), cv2.multiply(manual_frame, np.full(manual_frame.shape[-1], .8)).astype(np.uint8))
    if manual_group_track is not None:
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

def split_groups(frame, background_sub, NUM_BEES, groups, prev_tracks):
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
            point_colors.append(frame[point[1], point[0]])

        # Threshold based on the colors of the points
        color_thresh = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        for point_color in point_colors:
            color_thresh = cv2.bitwise_or(color_thresh, color_threshold(frame, point_color, MAX_THRESH_COLOR_DIFF))
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
            bee_contours = form_contours(edges, MIN_BEE_AREA, MAX_BEE_AREA, NUM_BEES=NUM_BEES, remove_background=True, remove_extra_contours=False)
            if len(bee_contours) == len(group_track["labels"]):
                correct_num_bees = True
                cv2.destroyWindow(MANUAL_WINDOW_NAME)

            cv2.drawContours(visualization, bee_contours, -1, (0, 255, 0), 2)
            for point in points:
                cv2.circle(visualization, point, 5, (0, 0, 255), -1)
            visualization = cv2.bitwise_or(cv2.multiply(mask, np.full(mask.shape[-1], .4)), cv2.multiply(visualization, np.full(mask.shape[-1], .7)))

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
                    color_thresh = cv2.bitwise_or(color_thresh, color_threshold(frame, point_color, MAX_THRESH_COLOR_DIFF - iteration))
                color_thresh = cv2.cvtColor(color_thresh, cv2.COLOR_GRAY2BGR)
                mask = np.copy(cv2.bitwise_and(processed, color_thresh).astype(np.uint8))
                iteration += 1

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