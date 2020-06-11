###### IMPORTS ######
# General
import os
import sys
import cv2
import glob
import json
import shutil
import subprocess
import argparse
import numpy as np

# Import other python files
import utils.general as general_utils
import utils.image as image_utils
import utils.ffmpeg as ffmpeg_utils

import modules.user_interaction as user_interaction
import modules.hyper_param_search as hyper_param_search

COLORS = {
    "individual"         : (0,255,0),
    "cluster"            : (138,43,226),
    "cluster_individual" : (0,255,0),
    "junk"               : (255,0,0)
}

def create_helper_dirs(data_root, dirname):
    # Create Directories
    drawn_frames_dir = general_utils.make_directory(dirname, root=data_root, remove_old=True)

    return drawn_frames_dir

def read_in_frames(denoised_frames_dir):
    denoised_filepaths = np.sort(glob.glob(f'{denoised_frames_dir}/**/denoised*/*.png', recursive=True))
    return denoised_filepaths

def filter_by_area(img, stats, min_area, max_area):
    areas = stats[:,-1]

    if min_area is None:
        min_area = np.min(areas)
    if max_area is None:
        max_area = np.max(areas)

    condition_1 = areas > min_area
    condition_2 = areas < max_area

    filter_idxs = np.logical_and(condition_1, condition_2)
    return filter_idxs

def run_group_filtering(denoised_img, filtered_stats, min_area, max_area, max_permitable_detection_area):
    # Mask image
    img_mask = image_utils.adaptive_thresholding(denoised_img)

    first_found = False
    for stat_i, (x, y, w, h, a) in enumerate(filtered_stats):

        bbox_area = w*h
        if bbox_area > max_permitable_detection_area:
            continue

        # GLOBAL: Compute block mask (1's where group bbox is; 0's everywhere else)
        group_mask = np.zeros_like(img_mask)
        group_mask[y:y+h,x:x+w] = 1

        # GLOBAL: Apply group mask to full image mask (make all pixels 0 where group is not, 1s where group is)
        masked_group_img = img_mask*group_mask

        # LOCAL: Crop out group mask
        cropped_group_img = img_mask[y:y+h,x:x+w]

        # LOCAL: Create mask by adaptive thresholding + morphological opening
        cropped_group_mask = image_utils.adaptive_filter_plus_opening(cropped_group_img, kernel_dim=(11,11))

        # LOCAL: Run connected components on group mask
        num_regions, regions, stats, centroids = cv2.connectedComponentsWithStats(cropped_group_mask)

        # Readjust local coords -> global coords
        global_stats_i = np.zeros_like(stats)
        for stat_i, stat in enumerate(stats):
            new_x, new_y, new_w, new_h, new_a = stat
            global_stats_i[stat_i] = np.array([x+new_x, y+new_y, new_w, new_h, new_a])

        if not first_found:
            total_global_stats = global_stats_i
            first_found = True
        else:
            total_global_stats = np.concatenate([total_global_stats, global_stats_i], axis=0)

        # Filter out by area
        area_filter_idxs = filter_by_area(denoised_img, total_global_stats, min_area, max_area)
        filtered_global_stats = total_global_stats[area_filter_idxs]

    return filtered_global_stats

def draw_group_on_frame(data_logger, denoised_img, draw_img, stats, group, min_area=None, max_area=None, box_width=1, draw_centroid=True):

    color = COLORS[group]

    total_img_area = int(np.product(denoised_img.shape[:2]))
    max_permitable_detection_area = total_img_area * 1/16

    # Get filter idxs
    if group == 'individual':
        min_area_i = min_area
        max_area_i = max_area
    elif group == 'cluster':
        min_area_i = max_area
        max_area_i = None
    elif group == 'junk':
        min_area_i = None
        max_area_i = min_area

    area_filter_idxs = filter_by_area(denoised_img, stats, min_area_i, max_area_i)

    # Filter the stats on area conditions
    filtered_stats = stats[area_filter_idxs]

    # Iterate over stats
    for stat_i, (x, y, w, h, a) in enumerate(filtered_stats):
        # Filter out detections with too large an area
        # -------------------------------------------------
        bbox_area = w*h
        if bbox_area > max_permitable_detection_area:
            continue
        # -------------------------------------------------

        # Filter out oblong detections
        # -------------------------------------------------
        skew_factor = 4
        bad_condition_1 = w > skew_factor*h
        bad_condition_2 = h > skew_factor*w
        if bad_condition_1 or bad_condition_2:
            continue
        # -------------------------------------------------

        data_logger = general_utils.log_data(data_logger, x,y,w,h,group)
        image_utils.draw_box(draw_img, x,y,w,h, box_width, color=color, draw_centroid=draw_centroid)

    # Check if cluster - try to separate out with thresholding + morphological opening
    if group == 'cluster':
        group_stats = run_group_filtering(denoised_img, filtered_stats, min_area, max_area, max_permitable_detection_area)

        for stat_i, (x, y, w, h, a) in enumerate(group_stats):
            bbox_area = w*h
            if bbox_area > max_permitable_detection_area:
                continue
            data_logger = general_utils.log_data(data_logger, x,y,w,h,'individual')
            color = COLORS['cluster_individual']
            image_utils.draw_box(draw_img, x,y,w,h, box_width, color=color, draw_centroid=draw_centroid)

def process_images(denoised_filepaths, drawn_frames_dir, min_area, max_area):
    data_logger = {}
    for denoised_img_i, denoised_filepath in enumerate(denoised_filepaths):
        if args.limit and denoised_img_i > args.limit:
            break

        sys.stdout.write(f'\rProcessing frame {denoised_img_i+1}/{len(denoised_filepaths)}')
        sys.stdout.flush()

        denoised_img = cv2.cvtColor(cv2.imread(denoised_filepath), cv2.COLOR_BGR2GRAY)

        # Frame id
        frame_id = os.path.splitext(os.path.basename(denoised_filepath))[0]

        # Initialize frame_id logger
        data_logger[frame_id] = []

        # Make single channel
        if len(denoised_img.shape) == 3:
            denoised_img = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)

        # Setup image for drawing on
        draw_img = general_utils.setup_draw_img(denoised_img)

        # Mask image
        img_mask = image_utils.adaptive_filter_plus_opening(denoised_img, invert=True)

        # Run Connected Components
        num_regions, regions, stats, centroids = cv2.connectedComponentsWithStats(img_mask)

        # Draw Individual detections
        draw_group_on_frame(data_logger[frame_id], denoised_img, draw_img, stats, group='individual',
                            min_area=min_area, max_area=max_area, box_width=5)

        # Draw Cluster detections
        if args.draw_clusters:
            draw_group_on_frame(data_logger[frame_id], denoised_img, draw_img, stats, group='cluster',
                                min_area=min_area, max_area=max_area, box_width=5)

        # Draw Trash detections
        if args.draw_trash:
            draw_group_on_frame(None, denoised_img, draw_img, stats, group='junk',
                                min_area=min_area, max_area=max_area, box_width=5)

        # Write draw image
        path = os.path.join(drawn_frames_dir, f"frame_{denoised_img_i+1:05d}.png")
        cv2.imwrite(path, draw_img[...,::-1])
    return data_logger

def main(UI, args):
    # Select the video
    print("-- Select video from list...")
    src_processed_root = general_utils.select_file(args.data_root)

    # Create helper dirs
    print("Setting up helper directories...")
    drawn_frames_dir = create_helper_dirs(src_processed_root, dirname='detection_frames')

    # Read in frames
    print("\n-- Reading in frames...")
    denoised_filepaths = read_in_frames(src_processed_root)

    # User interface for selecting proper stuff...
    # Pick an image somewhere in middle of video - x seconds in
    img_i = args.img_i
    denoised_img_i = cv2.cvtColor(cv2.imread(denoised_filepaths[img_i]), cv2.COLOR_BGR2GRAY)

    # Setup UI
    UI.run(denoised_img_i, src_processed_root, load_previous=args.load_prev_UI_results)
    min_area, max_area = UI.min_area, UI.max_area

    # Process the images
    data_logger = process_images(denoised_filepaths, drawn_frames_dir, min_area, max_area)
    print("\n")

    # Log data
    logger_path = os.path.join(src_processed_root, 'data_log.json')
    with open(logger_path, 'w') as outfile:
        json.dump(data_logger, outfile)

    ffmpeg_utils.frame2vid(drawn_frames_dir, src_processed_root, args)

def setup_args():
    parser = argparse.ArgumentParser(description='Process bee videos!')
    parser.add_argument('--i', dest='data_root', type=str, default='data/processed',
                        help='Set path to processed video frames')
    parser.add_argument('-r', '--fps', dest='FPS', type=float, default=25,
                        help='Frames per second (FPS)')
    parser.add_argument('--limit', dest='limit', type=int, default=0,
                        help='Processing limit')
    parser.add_argument('--v', dest='verbose', type=bool, default=False,
                        help='FFMPEG Verbosity')
    parser.add_argument('--f', dest='force', type=bool, default=True,
                        help='Force overwrite: True/False')
    parser.add_argument('--clusters', dest='draw_clusters', type=bool, default=True,
                        help='Draw cluster detections')
    parser.add_argument('--trash', dest='draw_trash', type=bool, default=False,
                        help='Draw trash detections')
    parser.add_argument('--prevUI', dest='load_prev_UI_results', type=bool, default=False,
                        help='Use previous UI results?')

    parser.add_argument('--img_idx', dest='img_i', type=int, default=1, help='Image index to label for this video')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    print("\n---------- Detecting bees ----------")
    args = setup_args()
    UI = hyper_param_search.UserInterface()
    main(UI, args)
    print("\n")
