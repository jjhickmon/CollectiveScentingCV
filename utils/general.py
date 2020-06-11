import os
import glob
import subprocess
import shutil
import numpy as np

def setup_draw_img(base_img):
    draw_img = base_img.copy()
    if len(draw_img.shape) == 2:
        draw_img = np.tile(draw_img[...,np.newaxis], (1,1,3))
    elif draw_img.shape[-1] == 1:
        draw_img = np.tile(draw_img, (1,1,3))
    return draw_img

def compute_centroid(x,y,w,h):
    centroid_x = x + w/2
    centroid_y = y + h/2
    return (int(centroid_x), int(centroid_y))

def make_directory(dirname, root=os.getcwd(), remove_old=False):
    dirname = os.path.sep.join(dirname.split("/"))
    new_dir = os.path.join(root, dirname)
    if not os.path.exists(new_dir):
        print(f"\nCreating '{new_dir}'")
        os.makedirs(new_dir)
    else:
        if remove_old:
            print(f"Path '{new_dir}' exists! Recreating it.")
            shutil.rmtree(new_dir)
            os.makedirs(new_dir)
    return new_dir

def log_data(data_logger, x,y,w,h,group):
    if data_logger is not None:
        data = {
            "x"  : float(x),
            "y"  : float(y),
            "h"  : float(h),
            "w"  : float(w),
            "id" : group
        }
        data_logger.append(data)

    return data_logger

def select_file(src_video_root, max_failed_attempts=3, prefix=''):
    src_video_paths = glob.glob(f'{src_video_root}/*{prefix}')

    for src_video_path_i, src_video_path in enumerate(src_video_paths):
        print(f"{src_video_path_i} : {os.path.basename(src_video_path)}")

    num_attempts = 0
    while True:
        if num_attempts >= max_failed_attempts:
            print("\n**Too many failed attempts. Exiting program.")
            exit()

        num_attempts += 1

        user_input = input("Select video by index: ")
        try:
            user_input_idx = int(user_input)
        except:
            print("Enter valid index integer")
            continue
        else:
            if user_input_idx >= len(src_video_paths):
                print(f"Invalid index choice. Only {len(src_video_paths)} videos exist to chose from.")
                continue
            else:
                src_video_path = src_video_paths[user_input_idx]
                break
    return src_video_path
