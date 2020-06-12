###### IMPORTS ######
# General
import os
import sys
import cv2
import glob
import json
import argparse
import pandas as pd
import numpy as np

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.models
import torchvision.transforms as transforms

# Import other python files
import orientation_estimation.modules.Utils as Utils
import orientation_estimation.modules.DataHandler as DataHandler
import orientation_estimation.modules.DataSamplers as DataSamplers
import orientation_estimation.modules.EvaluationUtils as Evaluation

# sys.path.append('../')
import utils.general as general_utils

def build_resnet(num_classes=1):
    print(f"Building resnet-18 with {num_classes} class(es)...")
    resnet = torchvision.models.resnet18(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_classes),
    )
    return resnet

def predict(bee_data, data_loader, model, device):
    orientations = list(np.zeros(len(bee_data.data_df)))
    counter = 0
    try:
        for i, (X, y, label_scenting, experiment, frame, crop) in enumerate(data_loader):
            sys.stdout.write(f'\r Batch {i+1} / {len(data_loader)}')
            sys.stdout.flush()

            X = Utils.convert_X_for_resnet(X)
            X = X.to(device)
            y = y.to(device)

            logits = model(X)

            # Negative angles to positive (add 360 to degrees or (2*np.pi) to radians)
            preds = np.rad2deg(logits.cpu().detach().numpy())

            # Convert negative angles to positive
            mask = (preds < 0) * 360
            preds = preds + mask

            for img_i in range(args.batch_size):
                try:
                    orientations[counter] = preds[img_i].tolist()
                    counter += 1
                except:
                    pass
    except KeyboardInterrupt or IndexError:
        print('\nEnding early.')
    return orientations

def save_prediction(bee_data, orientation, folder_paths):
    bee_data.data_df['orientation'] = orientation
    labeled_h5 = bee_data.data_df.to_dict('list')
    resnet_save_path = os.path.join(folder_paths[0], 'data_log_orientation.json')
    with open(resnet_save_path, 'w') as outfile:
        json.dump(labeled_h5, outfile)
        outfile.close()

def setup_args():
    parser = argparse.ArgumentParser(description='Classify scenting bees!')
    parser.add_argument('-p', '--data_root', dest='data_root', type=str, default='data/processed')
    parser.add_argument('-m', '--model_file', dest='model_file', type=str, default='ResnetOrientationModel.pt')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('-c', '--num_classes', dest='num_classes', type=int, default=1)
    args = parser.parse_args()
    return args

def main(args):
    # Select the video
    print("-- Select video from list...")
    src_processed_root = general_utils.select_file(args.data_root)
    print(f'\nProcessing video: {src_processed_root}')

    print("\n---------- Estimating bee orientations ----------")
    # Obtain up paths for video folder
    vid_name = src_processed_root.split('/')[-1]
    folder_paths = glob.glob(f'{args.data_root}/{vid_name}*')
    json_paths = sorted([os.path.join(folder, f'data_log_scenting.json') for folder in folder_paths])
    frames_path = f'denoised_frames/'

    # ------------------------------------------------------------- #
    ##### DATASET & TRANSFORMS ######
    # Set default transforms
    baseline_transforms = transforms.Compose([transforms.ToTensor()])

    print(f'Setting up data handler...')
    bee_data = DataHandler.BeeDataset_2(args.data_root, json_paths, frames_path,
                          baseline_transforms, augment_transforms=None,
                          mode='eval')
    print(f'Number of bee images to process: {len(bee_data.data_df)}')

    # ------------------------------------------------------------- #
    ###### DATALOADER ######
    print(f'Setting up data loader...\n')
    data_loader = DataLoader(bee_data, batch_size=args.batch_size, drop_last=False)

    # ------------------------------------------------------------- #
    ###### MODEL ######
    model = build_resnet(args.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on device: {device}\n')

    # ------------------------------------------------------------- #
    ###### LOAD TRAINED MODEL ######
    print(f"Loading trained model...\n")
    load_path = f'orientation_estimation/saved_models/{args.model_file}'
    load_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(load_dict['model'])
    metrics = load_dict['metrics']
    model.eval()
    model.to(device);

    # ------------------------------------------------------------- #
    ###### RUN TRAINED MODEL ######
    print(f"Estimating bee body orientations...")
    orientations = predict(bee_data, data_loader, model, device)

    # ------------------------------------------------------------- #
    # ###### SAVED OUTPUT DATA ######
    print(f'\nSaving orientation estimations...')
    save_prediction(bee_data, orientations, folder_paths)
    print(f"Fin.")

if __name__ == '__main__':
    args = setup_args()
    main(args)
    print("\n")
