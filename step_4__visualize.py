'''
Make a movie visualizing the scenting classification &
orientation estimation data on frames to label scenting bees
and their scenting directions.
'''

###### IMPORTS ######
import os
import sys
import cv2
import glob
import glob2
import json
import argparse
import pandas as pd
import numpy as np
import subprocess

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks")
plt.rcParams["font.family"] = "Arial"

import utils.general as general_utils

###### HELPER FUNCTIONS ######
def get_wanted_frames(start_frame, end_frame, all_frames):
    start_num = int(start_frame.split('_')[1])
    end_num = int(end_frame.split('_')[1])
    wanted_frames = np.arange(start_num, end_num+1)
    all_frame_nums = np.array([int(os.path.basename(frame).split('_')[1].split('.')[0])
                               for frame in all_frames])
    condition_1 = np.in1d(all_frame_nums, wanted_frames)
    return all_frames[condition_1]

def get_endpoint(point, angle, length):
    '''
    point - Tuple (x, y)
    angle - Angle you want your end point at in degrees.
    length - Length of the line you want to plot.
    '''
    # Unpack the first point
    x, y = point
    # Find the end point
    endy = y + (length * np.sin(np.radians(angle)))
    endx = x + (length * np.cos(np.radians(angle)))
    return endx, endy

def imgs2vid(imgs, outpath, fps):
    ''' Stitch together frame imgs to make a movie. '''
    height, width, layers = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(outpath, fourcc, fps, (width, height), True)

    for img_i, img in enumerate(imgs):
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

def setup_args():
    parser = argparse.ArgumentParser(description='Visualize results!')
    parser.add_argument('-p', '--data_root', dest='data_root', type=str, default='data/processed')
    parser.add_argument('-r', '--fps', dest='fps', type=int, default=15)
    args = parser.parse_args()
    return args

###### MAIN ######
def main(args):
    #------- SET FILE PATHS -------#
    # Select the video
    print("-- Select video from list...")
    src_processed_root = general_utils.select_file(args.data_root)
    print(f'\nProcessing video: {src_processed_root}')

    # Obtain up paths for video folder
    vid_name = src_processed_root.split('/')[-1]
    folder_paths = glob.glob(f'{args.data_root}/{vid_name}*')
    json_paths = sorted([os.path.join(folder, f'data_log.json') for folder in folder_paths])
    frames_path = f'denoised_frames/'

    #------- PROCESS DATA -------#
    # Open json data from neural net model evaluation
    resnet_save_path = os.path.join(folder_paths[0], 'data_log_orientation.json')
    with open(resnet_save_path, 'r') as infile:
        resnet_data = json.load(infile)

    # Make data into dataframe
    resnet_df = pd.DataFrame(resnet_data)

    # Obtain first and last frames
    start_frame = resnet_df['frame_num'][0]
    end_frame = resnet_df['frame_num'][len(resnet_df['frame_num'])-1]

    # Get path of all frames & make folder to store labeled frames
    frames_path = os.path.join(folder_paths[0], f'denoised_frames/')
    all_frames = np.sort(glob.glob(f'{frames_path}frame_*.png'))
    all_wanted_frames = get_wanted_frames(start_frame, end_frame, all_frames)
    resnet_frames_path = os.path.join(folder_paths[0], f'output_frames/')
    os.makedirs(resnet_frames_path, exist_ok=True)

    #------- VISUALIZE SCENTING DIRECTIONS ON SCENTING BEES -------#
    # Loop over frames, plot, save frame
    height = 0
    width = 0
    for frame_i, frame_path in enumerate(all_wanted_frames):
        sys.stdout.write(f'\rDrawing frame {frame_i+1}/{len(all_wanted_frames)}')
        sys.stdout.flush()

        frame_num = frame_path.split('/')[-1].split('_')[-1].split('.')[0]

        frame_img = cv2.imread(frame_path)
        height = frame_img.shape[0]
        width = frame_img.shape[1]

        # Loop over scenting bees for frame
        # Find scenting bees in df
        condition_1 = resnet_df['frame_num'] == 'frame_' + frame_num
        all_idxs = resnet_df.index[condition_1].tolist()

        condition_2 = resnet_df['classification'] == 'scenting'
        scenting_idxs = resnet_df.index[condition_1 & condition_2].tolist()

        # Draw orientation arrow on scenting bees
        for i in scenting_idxs: # all_idxs:
            x = resnet_df.at[i, 'x']    # Top left corner of bounding box
            y = resnet_df.at[i, 'y']    # Top left corner of bounding box
            w = resnet_df.at[i, 'w']    # Width of bounding box
            h = resnet_df.at[i, 'h']    # Height of bounding box
            orientation = resnet_df.at[i, 'orientation'][0]

            # Compute centroid from top left corner and w, h
            centroid_x = x + (w/2)
            centroid_y = y + (h/2)

            # ========== Plot orientation arrow
            length = [35 if i in scenting_idxs else 20][0]
            thickness = [5 if i in scenting_idxs else 4][0]
            scenting_color = (3, 252, 244)
            non_scenting_color = (31, 194, 91)
            arrow_color = [scenting_color if i in scenting_idxs else non_scenting_color][0]

            # Centroid is midpoint, arrow points:
            midpoint_x = centroid_x
            midpoint_y = centroid_y
            pred_endx, pred_endy = get_endpoint((midpoint_x, midpoint_y), angle=-int(orientation), length=length//2)
            pred_x1, pred_y1 = get_endpoint((midpoint_x, midpoint_y), angle=-int(orientation)+180, length=length//2)

            # Plot arrows
            cv2.arrowedLine(frame_img, (int(pred_endx), int(pred_endy)),
                            (int(pred_x1), int(pred_y1)), color=arrow_color,
                            thickness=thickness, tipLength=0.6)

        # Write frame to file
        save_path = f'{resnet_frames_path}/frame_{frame_num}.png'
        cv2.imwrite(save_path, frame_img[...,::-1])

    #------- MAKE MOVIE FROM FRAMES -------#
    print('\nMaking movie...')
    all_img_paths = np.sort(glob2.glob(f"{folder_paths[0]}/output_frames/*.png"))
    all_imgs = np.array([cv2.imread(img) for img in all_img_paths])
    save_title = 'output_movie'
    savepath = f'{folder_paths[0]}/{save_title}.mp4'
    imgs2vid(all_imgs, savepath, args.fps)
    print('Fin.\n')

if __name__ == '__main__':
    args = setup_args()
    main(args)
