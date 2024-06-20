import os
import sys
import numpy as np
import cv2
import shutil
import glob
import imutils
import argparse
import matplotlib
matplotlib.use('TkAgg')

import modules.orientation_GUI
import utils.general as general_utils
import utils.image as image_utils
import utils.ffmpeg as ffmpeg_utils

def create_helper_dirs(data_root, src_video_path):
    raw_frames_dirname = 'raw_frames'
    denoised_frames_dirname = 'denoised_frames'

    # Get video name
    video_name = os.path.splitext(os.path.basename(src_video_path))[0]

    # Create Directories
    processed_dir = general_utils.make_directory(video_name, root=data_root)
    raw_frames_dir = general_utils.make_directory(raw_frames_dirname, root=processed_dir)
    _ = general_utils.make_directory(denoised_frames_dirname, root=processed_dir)

    return raw_frames_dir, denoised_frames_dirname

def pad_image(img, pad_val=50):
    pad_shape = list(np.array(img.shape[:2]) + pad_val*2)
    if len(img.shape) == 3:
        pad_shape += [img.shape[2]]
    padded_img = np.zeros(pad_shape, dtype=np.uint8)
    padded_img[pad_val:img.shape[0]+pad_val, pad_val:img.shape[1]+pad_val] = img
    return padded_img

def correct_orientation(raw_frames_dir):
    frame_paths = np.sort(glob.glob(f'{raw_frames_dir}/frame_*.png'))

    init_frame = pad_image(cv2.imread(frame_paths[0])[...,::-1])
    gui = modules.orientation_GUI.Visualizer(init_frame, compute_angle=True)

    for frame_i, frame_path in enumerate(frame_paths):
        if frame_i >= 0:
            try:
                sys.stdout.write(f'\rPath {frame_i+1}/{len(frame_paths)}')
                sys.stdout.flush()
                frame_i = cv2.imread(frame_path)
                new_img = imutils.rotate_bound(frame_i, gui.degree)
                cv2.imwrite(frame_path, new_img)
            except:
                continue

def tight_crop(raw_frames_dir):
    frame_paths = np.sort(glob.glob(f'{raw_frames_dir}/frame_*.png'))

    init_frame = cv2.imread(frame_paths[0])[...,::-1]
    gui = modules.orientation_GUI.Visualizer(init_frame, compute_angle=False)

    for frame_i, frame_path in enumerate(frame_paths):
        try:
            sys.stdout.write(f'\rPath {frame_i+1}/{len(frame_paths)}')
            sys.stdout.flush()
            frame_i = cv2.imread(frame_path)
            new_img = frame_i[gui.points['y1']:gui.points['y2'], gui.points['x1']:gui.points['x2']]
            cv2.imwrite(frame_path, new_img)
        except:
            continue

def denoise_frames(raw_frames_dir, denoised_frames_dirname):
    raw_frame_paths = np.sort(glob.glob(f'{raw_frames_dir}/*.png'))
    for path_i, raw_frame_path in enumerate(raw_frame_paths):
        try:
            sys.stdout.write(f'\rPath {path_i+1}/{len(raw_frame_paths)}')
            sys.stdout.flush()
            raw_frame_img = cv2.imread(raw_frame_path)

            denoised_img = image_utils.remove_lines_BGR(raw_frame_img)

            denoised_path = raw_frame_path.replace(os.path.basename(raw_frames_dir), denoised_frames_dirname)
            cv2.imwrite(denoised_path, denoised_img)
            del raw_frame_img # Free up memory
            del denoised_img # Free up memory
        except:
            print("Error: Could not denoise frame. Skipping...")
            continue

def main(args):
    # Select the video
    print("Selecting Video...")
    video_dir = general_utils.select_file(args.data_root)
    src_video_path = general_utils.select_file(video_dir, prefix='.mp4')

    # Create helper dirs
    print("Setting up Helper Directories...")
    raw_frames_dir, denoised_frames_dirname = create_helper_dirs(args.output_root, src_video_path)
    # raw_frames_dir = "data/processed/C0116_short/raw_frames"
    # denoised_frames_dirname = "denoised_frames"

    # Convert video 2 frames
    # NOTE: For longer videos, use jpg to save space, has loss but is faster
    print("Converting Video to Frames...")
    ffmpeg_utils.vid2frames_simple(src_video_path, raw_frames_dir, args)
    print("Number of frames:", len(glob.glob(f'{raw_frames_dir}/*.png')))

    # Correct orientation
    # NOTE: Drow line along the top of the frame, press 'q' when done
    print("Realigning Frames...")
    correct_orientation(raw_frames_dir)

    # Slice video
    # NOTE: Draw line along the diagonal of the frame, press 'q' when done
    print("Cropping Frames...")
    tight_crop(raw_frames_dir)

    # Denoise frames
    print("Denoising Images...")
    denoise_frames(raw_frames_dir, denoised_frames_dirname)

    # Export video
    print("Exporting Video...")
    denoised_paths = np.sort(glob.glob(f'{raw_frames_dir}/*.png'))
    frame = cv2.imread(denoised_paths[0])
    height, width = len(frame), len(frame[0])
    path = src_video_path.replace('.mp4', '_processed.mp4')
    video = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for denoised_path in denoised_paths:
        img = cv2.imread(denoised_path)
        video.write(img)
        del img # Free up memory
    video.release()
    print(f'Video exported to {path}')

    if args.cleanup:
        print("Cleaning up raw frames...")
        shutil.rmtree(raw_frames_dir)
    print("Fin.")

def setup_args():
    parser = argparse.ArgumentParser(description='Process bee videos!')
    parser.add_argument('-i', '--in-root', dest='data_root', type=str, default='data/processed',
                        help='Set path to video files')
    parser.add_argument('-o', '--out-root', dest='output_root', type=str, default='data/processed',
                        help='Set processed output path')
    parser.add_argument('-r', '--fps', '--frames-per-second', dest='FPS', type=float, default=25,
                        help='Frames per second (FPS)')
    parser.add_argument('-s', '--ss', dest='start_second', type=int, default=None,
                        help='Second to start on')
    parser.add_argument('-t', '--to', dest='end_second', type=int, default=None,
                        help='Second to stop on')
    parser.add_argument('-f', '--force', dest='force', type=str, default='True',
                        help='Force overwrite: True/False')
    parser.add_argument('-v', '--verbose', dest='verbose', type=str, default='False',
                        help='FFMPEG Verbosity: True/False')
    parser.add_argument('-c', '--cleanup', dest='cleanup', type=str, default='False',
                        help='Cleanup raw files')
    parser.add_argument('-d', '--debug', dest='debug', type=str, default='False',
                        help='Debug mode: True/False')

    args = parser.parse_args()

    negative_keys = ['f', 'false', 'False', 'F', '0']
    args.force = args.force not in negative_keys
    args.cleanup = args.cleanup not in negative_keys
    args.verbose = args.verbose not in negative_keys
    args.debug = args.debug not in negative_keys

    if args.debug:
        args.start_second = 0
        args.end_second = 5
        args.FPS = 5

    return args

if __name__ == '__main__':
    args = setup_args()
    main(args)
