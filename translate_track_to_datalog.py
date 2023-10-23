import csv
import json
import argparse
import cv2
import os
from tqdm import tqdm
import math
import utils.general as general_utils
import utils.image as image_utils
import utils.ffmpeg as ffmpeg_utils
import numpy as np
import glob

def create_helper_dirs(data_root, dirname):
    # Create Directories
    drawn_frames_dir = general_utils.make_directory(
        dirname, root=data_root, remove_old=True)

    return drawn_frames_dir


def read_in_frames(denoised_frames_dir):
    denoised_filepaths = np.sort(
        glob.glob(f'{denoised_frames_dir}/**/denoised*/*.png', recursive=True))
    return denoised_filepaths

def vid2frames(src, output_path):
    vidcap = cv2.VideoCapture(src)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, image = vidcap.read()
    count = 0
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with tqdm(total=length) as pbar:
        while success:
            cv2.imwrite(f"{output_path}frame_{count:05d}.png", image)
            success, image = vidcap.read()
            count += 1
            pbar.update(1)

def visualize(data_log):
    drawn_frames_dir = create_helper_dirs(
        src_processed_root, dirname='detection_frames')
    denoised_filepaths = read_in_frames(src_processed_root)
    for frame_i, (denoised_filepath, key) in enumerate(zip(denoised_filepaths, data_log.keys())):
        denoised_img = cv2.cvtColor(cv2.imread(
            denoised_filepath), cv2.COLOR_BGR2GRAY)
        if len(denoised_img.shape) == 3:
            denoised_img = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)
        draw_img = general_utils.setup_draw_img(denoised_img)
        for bee in data_log[key]:
            x = bee["x"]
            y = bee["y"]
            w = bee["w"]
            h = bee["h"]
            image_utils.draw_box(draw_img, x, y, w, h, 1,
                                 color=(0, 255, 0), draw_centroid=True)
        path = os.path.join(drawn_frames_dir, f"frame_{frame_i+1:05d}.png")
        cv2.imwrite(path, draw_img[..., ::-1])
    ffmpeg_utils.frame2vid(drawn_frames_dir, src_processed_root, args)

def translate(tracking_file, output_file):
    translation = {}
    with open(tracking_file, mode='r') as file:
        # reading the CSV file
        csvFile = csv.reader(file)

        # displaying the contents of the CSV file
        for lines in csvFile:
            if lines[0] == "frame":
                continue
            frame = f"frame_{int(lines[0])+1:05d}"
            label = lines[2]
            x = float(lines[5])
            y = float(lines[6])
            angle = float(lines[14])
            lenMajor = float(lines[19]) + 20  # Attempting to account for extra distance from the wings
            # lenMinor = float(lines[21]) + 8
            rad1 = int(math.ceil(lenMajor / 2))

            boxX = int(x - rad1)
            boxY = int(y - rad1)
            # NOTE: Might need to change to draw rectangle
            boxH = int(rad1 * 2)
            boxW = int(rad1 * 2)
            is_merged = (lines[4].lower() == "true")

            bee_data = {"label": label, "x": boxX, "y": boxY, "h": boxH, "w": boxW, "angle": angle,
                        "id": "individual" if not is_merged else "cluster"}
            if frame in translation:
                translation[frame].append(bee_data)
            else:
                translation[frame] = [bee_data]

    with open(output_file, "w") as outfile:
        json.dump(translation, outfile)
    return translation


def setup_args():
    parser = argparse.ArgumentParser(description='Process bee videos!')
    parser.add_argument('-p', '--data_root', dest='data_root', type=str, default='data/processed',
                        help='Set path to processed video frames')
    parser.add_argument('-r', '--fps', dest='FPS', type=float, default=25,
                        help='Frames per second (FPS)')
    parser.add_argument('-v', '--verbose', dest='verbose', type=bool, default=False,
                        help='FFMPEG Verbosity')
    parser.add_argument('-f', '--force', dest='force', type=bool, default=True,
                        help='Force overwrite: True/False')
    parser.add_argument('-s', '--start_second', dest='start_second', type=int, default=None,
                        help='Start second of video to process')
    parser.add_argument('-e', '--end_second', dest='end_second', type=int, default=None,
                        help='End second of video to process')

    parser.add_argument('--img_idx', dest='img_i', type=int,
                        default=1, help='Image index to label for this video')

    args = parser.parse_args()

    return args

visualize_data = True
if __name__ == '__main__':
    print("\n---------- Detecting bees ----------")
    args = setup_args()
    src_processed_root = general_utils.select_file(args.data_root)
    print("select video from list...")
    video = general_utils.select_file(src_processed_root)
    print("select tracking file from list...")
    tracking_file = general_utils.select_file(src_processed_root)
    vid2frames(video, f"{src_processed_root}/denoised_frames/")
    translation = translate(tracking_file, f"{src_processed_root}/data_log.json")
    # if visualize_data:
    #     visualize(translation)
