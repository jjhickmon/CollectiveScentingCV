import cv2
import numpy as np
import json
import utils.general as general_utils
import os

VIDEO_NAME = ""
BACKGROUND_NAME = ""
ALLOW_MANUAL_LABELLING = False
ALLOW_MANUAL_SEGMENTING = True
MAX_AUTOMATIC_ITERATIONS = 15
WINDOW_NAME = "frame"
MANUAL_WINDOW_NAME = "manual point select - press ENTER to continue"
FRAMES_PATH = "denoised_frames"
MIN_BEE_AREA = 30
MAX_BEE_AREA = 1500
# MIN_GROUP_AREA = 200
MAX_MOVE_DISTANCE = 100
MAX_GROUP_AREA = 5000
MAX_THRESH_COLOR_DIFF = 20
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255,255,0), (0,255,255), (255,0,255), (255,255,255)]

NUM_BEES = 4 # NOTE: Important: update this value
TEST = False

def imgs2vid(imgs, outpath, fps):
    ''' Stitch together frame imgs to make a movie. '''
    height, width, layers = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(outpath, fourcc, fps, (width, height), True)

    for img_i, img in enumerate(imgs):
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

def preprocess(frame, background):
    n = float(1.1)
    img = cv2.multiply(frame, np.array([n]))
    img_background = cv2.multiply(background, np.array([n]))

    # Background Subtraction
    background_sub = cv2.cvtColor(cv2.absdiff(img, img_background), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(background_sub, (25, 25), 0) # blur to quickly denoise, precision is not important for this step
    # multiplied = cv2.multiply(blurred, np.array([2.0]))
    _, threshold = cv2.threshold(blurred,thresh=45,maxval=255,type=cv2.THRESH_BINARY)
    queen_cage = [np.array([[493, 35], [555, 35], [570, 410], [508, 410]])]
    threshold = cv2.fillPoly(threshold, queen_cage, 0) # queen cage, points are clockwise from top left

    # opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
    # cv2.waitKey(0)
    closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=8)
    erode = cv2.erode(closing, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    dilate = cv2.dilate(erode, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations=3)
    # closing2 = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    if TEST:
        dilate = cv2.dilate(dilate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)

    dilate = cv2.fillPoly(dilate, queen_cage, 0) # queen cage, points are clockwise from top left
    frame = cv2.fillPoly(frame, queen_cage, 0) # queen cage, points are clockwise from top left
    # cv2.rectangle(closing, (485, 35, 90, 375), 0, -1) # queen cage
    cv2.rectangle(dilate, (410, 1000, 50, 100), 0, -1) # random artifact
    cv2.rectangle(dilate, (1200, 0, 300, 30), 0, -1) # random artifact
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
    # bee_contours_idx = []
    # for i, element in enumerate(hierarchy[0]):
    #     if element[2] == -1: # If no child, i.e. innermost contour
    #         bee_contours_idx.append(i)
    # bee_contours = [unmerged_contours[i] for i in bee_contours_idx]
    if remove_background:
        contours = [contour for i, contour in enumerate(contours) if hierarchy[i][2] == -1]
    # for contour in contours:
    #     if cv2.contourArea(contour) < MIN_GROUP_AREA:
    #         pass
    #         # print(cv2.contourArea(contour))
    # print("contours before area filter", len(contours))

    prev_num_contours = len(contours)
    contours = [contour for contour in contours if minArea < cv2.contourArea(contour) < maxArea]
    if not prev_num_contours == len(contours):
        print("Warning: Removed contours based on area")

    # print("contours after area filter", len(contours))
    if remove_extra_contours:
        if len(contours) > NUM_BEES:
            diff = len(contours) - NUM_BEES
            contours = sorted(contours, key=lambda contour: cv2.contourArea(contour))[diff:]
            print("Warning: Removed excess contours")
    return contours

def match_track_labels(tracks, prev_tracks):
    print("prev", [final_track["labels"] for final_track in prev_tracks] if prev_tracks is not None else [], "tracks", [final_track["labels"] for final_track in tracks])
    cv2.waitKey(0)
    # labels = [] # NOTE: test
    if prev_tracks is not None and (len(prev_tracks) > len(tracks) or len(prev_tracks) < len(tracks)): # if a group has been formed, i.e. no longer 1:1 matching
        print("matching tracks", len(tracks), len(prev_tracks))
        checked = []
        num_labels = 0
        for prev_track in prev_tracks:
            current_tracks = list(filter(lambda i: i not in checked, tracks)) # TODO: fix this
            while len(current_tracks) > 1:
                current_tracks = list(filter(lambda i: i not in checked, tracks))
                group_track = sorted(current_tracks, key=lambda track: cv2.norm(np.array(track["center"]) - np.array(prev_track["center"])))[0]
                if cv2.norm(np.array(group_track["center"]) - np.array(prev_track["center"])) < MAX_MOVE_DISTANCE:
                    group_track["labels"].extend(prev_track["labels"])

                    num_labels += 1
                    break
                else:
                    print("distance2")
                checked.append(group_track)
        if len(prev_tracks) < len(tracks):
            print("less")
            for track in tracks:
                if track not in checked:
                    track["labels"].append(num_labels)
                    num_labels += 1

            # labels.extend(prev_track["labels"])
    # TODO: match based on closest contour point not contour centers. More accurate for groups of more than one bee
    elif prev_tracks is not None and len(prev_tracks) == len(tracks): # if a group wasn't formed, i.e. 1:1 matching
        print("equal")
        matched = []
        for track in tracks:
            checked = []
            current_prev_tracks = list(filter(lambda i: i not in checked, prev_tracks))

            while len(current_prev_tracks) > 1:
                prev_tracks_new = filter(lambda i: i not in matched, prev_tracks)
                current_prev_tracks = list(filter(lambda i: i not in checked, prev_tracks_new))
                closest_track = sorted(current_prev_tracks, key=lambda prev_track: cv2.norm(np.array(prev_track["center"]) - np.array(track["center"])))[0]
                if cv2.norm(np.array(closest_track["center"]) - np.array(track["center"])) < MAX_MOVE_DISTANCE:
                    track["labels"] = closest_track["labels"]
                    print("test", track["labels"])
                    # labels.extend(track["labels"])
                    matched.append(closest_track)
                    break
                else:
                    print("distance")
                    checked.append(closest_track)
             # TODO: test new structure (uncomment this and remove the above for new structure)
            # closest_track = sorted(prev_tracks, key=lambda prev_track: cv2.norm(np.array(prev_track["center"]) - np.array(track["center"])))[0]
            # track["labels"] = closest_track["labels"]

    # elif prev_tracks is not None and len(prev_tracks) < len(tracks): # if a group wasn't split by watershed but now is separated
    #     # manually label
    #     print("greater")

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
        moment = cv2.moments(contour)
        if moment["m00"] == 0:
            continue # TODO: I'm not sure if I should skip this or not
        cX = int(moment["m10"] / moment["m00"])
        cY = int(moment["m01"] / moment["m00"])
        track_center = (cX, cY)

        # TODO: Maybe change initial labelling to be proximity to queen
        track = {"labels": [], "center": track_center, "contour": contour}
        tracks.append(track)

    # Assign Labels
    # TODO: test new structure (remove this for new structure)
    tracks = match_track_labels(tracks, prev_tracks)
    return tracks

def draw_tracks(frame, tracks, show_label=True, show_area=False):
    # Draw contours
    for track in tracks:
        if len(track["labels"]) > 1 or len(track["labels"]) == 0:
            color = (180, 180, 180)
        else:
            color = COLORS[track["labels"][0] % len(COLORS)]
        cv2.drawContours(frame, [track["contour"]], -1, color, 2)

        x,y,w,h = cv2.boundingRect(track["contour"])
        bottom_right = (x+w, y+h)
        track_text = ""
        if show_label:
            track_text = track_text + str(track["labels"])
        if show_area:
            track_text = track_text + str(cv2.contourArea(track["contour"]))
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
    labels = {}
    label = -1

    update_manual_frame(frame, processed, background_sub, points, manual_group_track)
    while(1):
        # mask = np.copy(dilate)
        cv2.setMouseCallback(MANUAL_WINDOW_NAME, on_click, [frame, processed, background_sub, points, label, manual_group_track])
        cv2.imshow(MANUAL_WINDOW_NAME, manual_frame)
        k = cv2.waitKey(30) & 0xFF
        # print("key pressed: ", k)
        if ALLOW_MANUAL_LABELLING:
            for num_key_code in range(48,58):
                if k == num_key_code:
                    label = k - 48
            # for i in range()
        if k == 2: # left arrow
            MAX_THRESH_COLOR_DIFF = MAX_THRESH_COLOR_DIFF - 1
            print("MAX_THRESH_COLOR_DIFF=",MAX_THRESH_COLOR_DIFF)
            # mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
            # print("erode")
        elif k == 3: # right arrow
            MAX_THRESH_COLOR_DIFF = MAX_THRESH_COLOR_DIFF + 1
            print("MAX_THRESH_COLOR_DIFF=",MAX_THRESH_COLOR_DIFF)
            # mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)
            # print("dilate")
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

    # cv2.imshow('done', manual_mask)
    # cv2.waitKey(0)
    return points, manual_mask

def split_groups(frame, background_sub, tracks, prev_tracks):
    visualization = np.copy(frame)
    tracks2 = []
    # Split groups
    # TODO: testing new structure
    for group_track in tracks:
        processed = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        processed = cv2.drawContours(processed, [group_track["contour"]], -1, (255, 255, 255), -1)

        assert group_track in tracks
        # Get the individual tracks that are apart of the group
        prev_unmerged_tracks = []
        # print("prev_tracks", len(prev_tracks), "tracks", len(tracks))
        for prev_track in prev_tracks:
            if not len(prev_track["labels"]) == 0 and set(prev_track["labels"]).issubset(group_track["labels"]):
                prev_unmerged_tracks.append(prev_track)
        # print("prev", len(prev_unmerged_tracks), len(group_track["labels"]), group_track["labels"])

        # cv2.drawContours(frame, [group_track["contour"]], -1, COLORS[0], 2)

        if not len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        # Get the colors of the bees in the group
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
            # cv2.imshow('frame5', visualization)
            # cv2.waitKey(0)
            for point in points:
                cv2.circle(visualization, point, 5, (0, 0, 255), -1)
            visualization = cv2.bitwise_or(cv2.multiply(mask, np.array([.4])), cv2.multiply(visualization, np.array([.7])))

            if iteration == MAX_AUTOMATIC_ITERATIONS:
                if not len(bee_contours) == len(group_track["labels"]) and ALLOW_MANUAL_SEGMENTING:
                    print("Error: Number of contours found is not ", len(group_track["labels"]), "number of contours found: ", len(bee_contours))
                    global_group_track = group_track
                    custom_points, custom_mask = manual_segmentation(frame, processed, background_sub, group_track)
                    reset_group = len(custom_points) == 0
                    points = custom_points
                    mask = custom_mask
                elif not len(bee_contours) == len(group_track["labels"]) and not ALLOW_MANUAL_SEGMENTING:
                    reset_group = True
            else:
                # print("iteration: ", iteration, "number of contours found: ", len(bee_contours))
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

        # print("correct_num_bees=", correct_num_bees, "reset_group=", reset_group)
        if correct_num_bees and not reset_group:
            separated_tracks = []
            for bee_contour in bee_contours:
                # Match contour to track
                bee_moment = cv2.moments(bee_contour)
                if bee_moment["m00"] == 0:
                    continue
                cX = int(bee_moment["m10"] / bee_moment["m00"])
                cY = int(bee_moment["m01"] / bee_moment["m00"])
                bee_center = (cX, cY)

                # match = sorted(prev_unmerged_tracks, key=lambda prev_unmerged_track: cv2.norm(np.array(prev_unmerged_track["center"]) - np.array(bee_center)))[0]
                # unmerged_track = {"labels": match["labels"], "center": bee_center, "contour": bee_contour}
                separated_track = {"labels": [], "center": bee_center, "contour": bee_contour}
                separated_tracks.append(separated_track)
            # print(group_track["labels"], len(separated_tracks), len(prev_unmerged_tracks))
            # assert len(separated_tracks) == len(group_track["labels"]) == len(prev_unmerged_tracks)

            # TODO: testing new structure (comment this and uncomment below for new structure)
            tracks2 = match_track_labels(separated_tracks, prev_unmerged_tracks)
            if group_track in tracks: # TODO: double check if necessary
                tracks.remove(group_track)
                tracks.extend(tracks2)

            # if group_track in tracks: # TODO: double check if necessary
            #     tracks.remove(group_track)
            #     tracks.extend(separated_tracks)
        # frame2 = draw_tracks(frame, tracks2)
        # cv2.imshow('frame2', frame2)
        # cv2.waitKey(0)

    return visualization, tracks

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
        # print("start")
        if not success:
            break
        raw_frames.append(frame)
        processed, background_sub = preprocess(frame, background)

        # Pass in MAX_GROUP_AREA in case groups were formed
        contours = form_contours(processed, MIN_BEE_AREA, MAX_GROUP_AREA, remove_background=False, remove_extra_contours=True)
        print("contours", len(contours))
        test_frame = cv2.drawContours(frame, contours, -1, (255, 255, 255), 2)
        cv2.imshow("test", test_frame)
        # print("contours formed", len(contours))
        if len(contours) == 0:
            print("Warning: No contours were created")
            continue
        tracks = create_tracks(contours, prev_tracks)
        print("tracks", len(tracks))
        if len(tracks) == 0:
            print("Warning: No tracks were created")
            continue

        # Create groups
        # TODO: test new structure (comment this for new structure)
        groups = [track for track in tracks if len(track["labels"]) > 1]
        if len(groups) >= 1:
            print("group")
            # for group in groups:
            #     print(group["labels"])
            # print("")
        if ALLOW_MANUAL_SEGMENTING and prev_tracks is not None:
            # frame, split_tracks = split_groups(frame, background_sub, tracks, prev_tracks)
            # TODO: test new structure (uncomment above and comment below for new structure)
            frame, split_tracks = split_groups(frame, background_sub, tracks, groups)
        else:
            split_tracks = tracks
        print("split_tracks", len(split_tracks))
        split_tracks = match_track_labels(split_tracks, prev_tracks)
        # frame = draw_tracks(frame, split_tracks, show_area=True)
        frame = draw_tracks(frame, groups) # TODO: test new structure comment this and uncomment above for new structure

        # draw_tracks(frame, split_tracks)
        cv2.imshow(WINDOW_NAME, frame)

        annotated_frames.append(frame)
        data_log[f"frame_{len(annotated_frames):05d}"] = []
        for split_track in split_tracks:
            x,y,w,h = cv2.boundingRect(split_track["contour"])
            # group = False
            group = split_track in groups #TODO: test new structure (comment this and uncomment above for new structure)
            data = { "label": str(split_track["labels"]).replace('[','').replace(']',''), "x": x, "y": y, "h": h, "w": w, "id": "individual" if not group else "cluster" }
            data_log[f"frame_{len(annotated_frames):05d}"].append(data)
        if split_tracks is not None and not len(split_tracks) == 0: # TODO: check if necessary
            prev_tracks = split_tracks
        else:
            prev_tracks = tracks
        # print("updated prev_tracks", len(prev_tracks))

        # NOTE: Pause with space, exit with esc
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
    video_outpath = f"{src_processed_root}/{VIDEO_NAME}_tracked.mp4"
    imgs2vid(annotated_frames, video_outpath, 30)
    if not os.path.exists(f'{src_processed_root}/{FRAMES_PATH}'):
        os.mkdir(f'{src_processed_root}/{FRAMES_PATH}')
        for i, raw_frame in enumerate(raw_frames):
            cv2.imwrite(f'{src_processed_root}/{FRAMES_PATH}/frame_{i+1:05d}.png', raw_frame)
    cap.release()
    cv2.destroyAllWindows()
    os.system('say "your program has finished"') # NOTE: Just for testing
