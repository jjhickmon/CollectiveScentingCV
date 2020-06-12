import os
import cv2
import sys
import json
import glob
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class BeeDataset(Dataset):
    '''
    Always apply baseline transforms; augmentations optional.
    '''
    def __init__(self, root_path, baseline_transforms, augment_transforms=None, mode='train'):
        # Mode for usage of this class
        self.mode = mode

        # Paths
        self.root_path = root_path

        # Transforms
        self.baseline_transforms = baseline_transforms
        self.augment_transforms = augment_transforms

        # Setups
        self.store_imgs()
        self.prep_data()
        self.process_annotations()

    def process_annotations(self):
        self.annotations_df = pd.read_table(os.path.join(self.root_path, 'annotations.txt'), delim_whitespace=True)

    def store_imgs(self):
        self.img_paths = sorted(glob.glob(f"{os.path.join(self.root_path)}/*.png"))

        # Create collection of images
        self.all_imgs = []
        for path in self.img_paths:
            img = cv2.imread(path, 0)
            self.all_imgs.append(img)

    def prep_data(self):
        self.img_dim = self.all_imgs[0].shape[:2]
        self.img_c = self.all_imgs[0][..., np.newaxis].shape[-1]

    def __getitem__(self, idx):
        img = self.all_imgs[idx][..., np.newaxis]

        label_theta = self.annotations_df.iloc[idx]['theta']
        head_x = self.annotations_df.iloc[idx]['head_x']
        head_y = self.annotations_df.iloc[idx]['head_y']
        tail_x = self.annotations_df.iloc[idx]['tail_x']
        tail_y = self.annotations_df.iloc[idx]['tail_y']

        # Apply optinal augmentation transforms
        if self.augment_transforms is not None:
            img = self.augment_transforms(img)
        # For all, apply baseline transforms
        if self.baseline_transforms is not None:
            img = self.baseline_transforms(img)

        return img, label_theta, head_x, head_y, tail_x, tail_y

    def __len__(self):
        total_num_imgs = len(self.all_imgs)

##############################################################################

class BeeDataset_2(Dataset):
    '''
    Read and join jsons data, pad frames + crop bees on the fly.
    Always apply baseline transforms; augmentations optional.

    Two modes: 1) train for training and 2) eval for evaluating
    '''
    def __init__(self, root_path, json_paths, frames_path,
                 baseline_transforms, augment_transforms=None, mode='eval'):
        # Mode for usage of this class
        self.mode = mode

        # Paths
        self.root_path = root_path
        self.json_paths = json_paths
        self.frames_path = frames_path

        # Transforms
        self.baseline_transforms = baseline_transforms
        self.augment_transforms = augment_transforms

        # Setups
        self.load_json(self.json_paths[0])
        self.store_padded_frames()  # Setup a dictionary that contains padded frames
        self.prep_data()            # To get image dimensions

    # -------------- Deal with json data -------------- #
    def load_json(self, json_path):
        with open(json_path) as infile:
            data = json.load(infile)

        # For evaluating, give all bees an initial orientation class of 0
        data_dict = {}
        for key, data in data.items():
            data_dict[key] = data
            num_detections = len(data)

        # Make initial orientation data
        orientation = np.zeros(num_detections, dtype=int)
        data_dict['orientation'] = orientation

        # Make dataframe
        data_df = pd.DataFrame(data_dict)
        data_df = data_df[data_df['bee_type']=='individual']

        ##### Evaluate on only scenting bees
        if self.mode == 'eval':
            data_df = data_df[data_df['classification']=='scenting']
        ##### Evaluate on only scenting bees

        data_df.reset_index(drop=True, inplace=True)
        self.data_df = data_df

    # -------------- Pad frame image and crop bees -------------- #
    def pad_img(self, img, padding):
        # Pad a frame image to safely crop bees on edges
        h, w = img.shape[:2]
        new_h = h + 2*padding
        new_w = w + 2*padding
        padded_img = np.zeros((new_h, new_w)) + np.median(img)
        padded_img[padding:h+padding, padding:w+padding] = img
        return padded_img

    def compute_centroid(self, idx, padding):
        top_left_x = self.data_df['x'][idx] + padding
        top_left_y = self.data_df['y'][idx] + padding
        height = self.data_df['h'][idx]
        width = self.data_df['w'][idx]
        centroid_x = top_left_x + (width/2)
        centroid_y = top_left_y + (height/2)
        return centroid_x, centroid_y

    def crop_img(self, padded_img, centroid_x, centroid_y, new_wh=(80,80)):
        # Using new_h, new_w, compute new x, y (top left) to crop
        w = new_wh[0]
        h = new_wh[1]
        top_left_x = int(centroid_x - (w/2))
        top_left_y = int(centroid_y - (h/2))
        cropped_img = padded_img[top_left_y:top_left_y+h, top_left_x:top_left_x+w]
        return cropped_img

    # -------------- Store padded frames in memory -------------- #
    def store_padded_frames(self, padding=40):
        experiment_frames = self.data_df.groupby(['experiment_name', 'frame_num']).size().reset_index().rename(columns={0:'count'})

        self.frames_dict = {experiment_frames.iloc[idx]['experiment_name']:{} for idx in range(len(experiment_frames))}

        for idx in range(len(experiment_frames)):
            # if idx % 500 == 0:
            #     sys.stdout.write(f'\rReading in frame {idx+1}...')
            #     sys.stdout.flush()

            # Get experiment and frame names
            experiment_name = experiment_frames.iloc[idx]['experiment_name']
            frame_num = experiment_frames.iloc[idx]['frame_num']

            # Pad that frame and store in dict
            # Read in denoised frame
            frame_img = os.path.join(self.root_path, experiment_name, self.frames_path, f'{frame_num}.png')
            img = cv2.imread(frame_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Pad frame
            padded_img = self.pad_img(img, padding=padding)
            padded_img = padded_img[..., np.newaxis]
            padded_img = padded_img.astype(np.uint8)

            self.frames_dict[experiment_name][frame_num] = padded_img

    # -------------- Get an image's dims -------------- #
    def prep_data(self, padding=40, new_wh=(80,80)):

        # Get a frame and crop 1 image to get img_dim and img_c
        experiment_idx = self.data_df['experiment_name'][0]
        frame_idx = self.data_df['frame_num'][0]

        # Get padded image from frames_dict
        padded_img = self.frames_dict[experiment_idx][frame_idx]

        # Compute centroid
        centroid_x, centroid_y = self.compute_centroid(idx=0, padding=padding)

        # Crop image here to have dims for labeling
        cropped_img = self.crop_img(padded_img, centroid_x, centroid_y, new_wh)
        self.img_dim = cropped_img.shape[0:2]
        self.img_c = cropped_img.shape[-1]

    # -------------- For dataloader to iterate over batches -------------- #
    def __getitem__(self, idx, padding=40):
        experiment_idx = self.data_df['experiment_name'].iloc[idx]
        frame_idx = self.data_df['frame_num'].iloc[idx]
        crop_idx = self.data_df['cropped_number'].iloc[idx]

        # --------- Get padded image & Crop bee on the fly --------- #
        # Get padded image from frames_dict
        padded_img = self.frames_dict[experiment_idx][frame_idx]

        # Compute centroid
        centroid_x, centroid_y = self.compute_centroid(idx, padding)

        # Crop image here to show for labeling
        cropped_img = self.crop_img(padded_img, centroid_x, centroid_y)

        # --------- Get labels & convert --------- #
        # Get string labels
        label_theta = self.data_df['orientation'].iloc[idx]
        label_scenting = self.data_df['classification'].iloc[idx]

        # --------- Apply transforms --------- #
        # Apply optional augmentation transforms
        # For now, it is applying to all
        if self.augment_transforms is not None:
            cropped_img = self.augment_transforms(cropped_img)

        # For all, apply baseline transforms
        if self.baseline_transforms is not None:
            cropped_img = self.baseline_transforms(cropped_img)

        if self.mode == 'train':
            return cropped_img, label_int
        elif self.mode == 'eval':
            return cropped_img, label_theta, label_scenting, experiment_idx, frame_idx, crop_idx

    def __len__(self):
        num_labeled_individual_detections = len(self.data_df)
        return num_labeled_individual_detections
