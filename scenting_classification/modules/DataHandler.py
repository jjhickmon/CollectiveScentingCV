import os
import glob
import cv2
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class BeeDataset(Dataset):
    '''
    Always apply baseline transforms; augmentations optional. Processes images already cropped.
    '''
    def __init__(self, root_path, baseline_transforms, augment_transforms=None, mode='train', train_idxs=None):
        # Mode for usage of this class
        self.mode = mode

        # Paths
        self.root_path = root_path

        # Transforms
        self.baseline_transforms = baseline_transforms
        self.augment_transforms = augment_transforms
        self.train_idxs = train_idxs

        # Setups
        self.store_imgs()
        self.create_lookup()
        self.prep_data()

    def create_lookup(self):
        # Create label_str -> label_int lookup from data_df
        self.label_to_int = {'unclassified': -1, 'non_scenting': 0, 'scenting': 1}
        self.int_to_label = {-1: 'unclassified', 0: 'non_scenting', 1: 'scenting'}

    def store_imgs(self):
        self.scenting_img_paths = sorted(glob.glob(f"{os.path.join(self.root_path, f'scenting')}/*.png"))
        self.non_scenting_img_paths = sorted(glob.glob(f"{os.path.join(self.root_path, f'non_scenting')}/*.png"))

        scenting_imgs = []
        scenting_labels = []
        for path in self.scenting_img_paths:
            # Img
            img = cv2.imread(path, 0)
            scenting_imgs.append(img)
            # Label
            label = 'scenting'
            scenting_labels.append(label)

        non_scenting_imgs = []
        non_scenting_labels = []
        for path in self.non_scenting_img_paths:
            # Img
            img = cv2.imread(path, 0)
            non_scenting_imgs.append(img)
            # Label
            label = 'non_scenting'
            non_scenting_labels.append(label)

        self.all_img_paths = self.scenting_img_paths + self.non_scenting_img_paths
        self.all_imgs = scenting_imgs + non_scenting_imgs
        self.all_labels = scenting_labels + non_scenting_labels

    def prep_data(self):
        self.img_dim = self.all_imgs[0].shape[:2]
        self.img_c = self.all_imgs[0][..., np.newaxis].shape[-1]

    def __getitem__(self, idx):
        img = self.all_imgs[idx][..., np.newaxis]
        label_str = self.all_labels[idx]
        label_int = self.label_to_int[label_str]

        # Apply optional augmentation transforms on train set
        if self.augment_transforms is not None:
            if idx in self.train_idxs:
                img = self.augment_transforms(img)

        # For all, apply baseline transforms
        if self.baseline_transforms is not None:
            img = self.baseline_transforms(img)

        return img, label_int

    def __len__(self):
        total_num_imgs = len(self.all_imgs)

#============================================================================================

class BeeDataset_2(Dataset):
    '''
    Read and join jsons data, pad frames + crop bees on the fly.
    Always apply baseline transforms; augmentations optional.

    Two modes: 1) train for training and 2) eval for evaluating
    '''
    def __init__(self, root_path, json_paths, frames_path,
                 baseline_transforms, augment_transforms=None, mode='train'):
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
        self.join_jsons()           # Create cumulative json
        self.create_lookup()        # Define lookup for string/int label
        self.store_padded_frames()  # Setup a dictionary that contains padded frames
        self.prep_data()            # To get image dimensions

    # -------------- Deal with json data -------------- #
    def join_jsons(self):
        # Create a cumulative data json from all labeled data jsons
        data_df = []
        for json_path in self.json_paths:
            df = self.load_json(json_path)
            data_df.append(df)

        # Concatenate single df's
        self.data_df = pd.concat(data_df, ignore_index=True)

    def load_json(self, json_path):
        with open(json_path) as infile:
            data = json.load(infile)

        # If 'training' mode, check for labeled data (if has 'classification' and 'cropped_number')
        # If 'eval' mode, give all bees a classification & cropped_number
        labeled_data = {}
        for frame_key, frame_detections in data.items():
            classified_detections = []
            for detection_i, detection in enumerate(frame_detections):

                # If getting data for training, only get the labeled
                if self.mode == 'train':
                    if detection.get('classification', False):
                        classified_detections.append(detection)

                # If evaluating, give data point a class & crop num
                elif self.mode == 'eval':
                    if detection.get('classification', False) == False:
                        detection['classification'] = 'unclassified'
                        detection['cropped_number'] = f'crop_{detection_i:04d}'
                        classified_detections.append(detection)

            if classified_detections:
                labeled_data[frame_key] = classified_detections

        # Extract data from json dictionary
        experiment_name = []
        frame_num = []
        x = []
        y = []
        w = []
        h = []
        bee_type = []
        classification = []
        cropped_number = []

        for frame, detections in labeled_data.items():
            for d_i, d in enumerate(detections):
                experiment_name.append(json_path.split('/')[-2])
                frame_num.append(frame)
                x.append(d['x'])
                y.append(d['y'])
                w.append(d['w'])
                h.append(d['h'])
                bee_type.append(d['id'])
                classification.append(d['classification'])
                cropped_number.append(d['cropped_number'])

        # Make dataframe
        data_dict = {
            "experiment_name": experiment_name,
            "frame_num": frame_num,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "bee_type": bee_type,
            "classification": classification,
            "cropped_number": cropped_number
        }

        data_df = pd.DataFrame(data_dict)
        data_df = data_df[data_df['bee_type']=='individual']
        data_df.reset_index(drop=True, inplace=True)
        return data_df

    def create_lookup(self):
        # Create label_str -> label_int lookup from data_df
        self.label_to_int = {'unclassified': -1, 'non_scenting': 0, 'scenting': 1}
        self.int_to_label = {-1: 'unclassified', 0: 'non_scenting', 1: 'scenting'}

    # -------------- Pad frame image and crop bees -------------- #
    def pad_img(self, img, padding=80):
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
        label_str = self.data_df['classification'].iloc[idx]

        # Using lookup, convert label_str -> int
        label_int = self.label_to_int[label_str]

        # --------- Apply transforms --------- #
        # Apply optional augmentation transforms
        # For now, it is applying to all
        if self.augment_transforms is not None:
            condition_1 = label_str == 'scenting'
            condition_2 = label_str == 'non_scenting'
            if condition_1 or condition_2:
                cropped_img = self.augment_transforms(cropped_img)

        # For all, apply baseline transforms
        if self.baseline_transforms is not None:
            cropped_img = self.baseline_transforms(cropped_img)

        if self.mode == 'train':
            return cropped_img, label_int
        elif self.mode == 'eval':
            return cropped_img, label_int, experiment_idx, frame_idx, crop_idx

    def __len__(self):
        num_labeled_individual_detections = len(self.data_df)
        return num_labeled_individual_detections
