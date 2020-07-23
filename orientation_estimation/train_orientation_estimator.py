###### IMPORTS ######
# General
import os
import sys
import cv2
import glob
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sns.set(style="ticks")
plt.rcParams["font.family"] = "Arial"

# Pytorch
import torch
import torchvision
import torch.nn as nn
import torchvision.models
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Import other python files
import modules.Utils as Utils
import modules.DataHandler as DataHandler
import modules.DataSamplers as DataSamplers
import modules.EvaluationUtils as Evaluation

np.random.seed(42)

def train_val_test_split(bee_data, num_imgs, test_split=0.2, val_split=0.1):
    dev_split = 1 - test_split
    train_split = 1 - val_split

    img_idxs = np.arange(0, len(bee_data.all_imgs), 1)
    np.random.shuffle(img_idxs)

    # Split to dev and test
    num_dev_examples = int(len(img_idxs) * dev_split)
    dev_idxs = img_idxs[:num_dev_examples]
    test_idxs = img_idxs[num_dev_examples:]

    # Split dev into train and val
    num_train_examples = int(len(dev_idxs) * train_split)
    train_idxs = dev_idxs[:num_train_examples]
    val_idxs = dev_idxs[num_train_examples:]

    # Make sure no duplicates & sum up: 0 and True
    check_1 = np.in1d(train_idxs, val_idxs).sum()
    check_2 = np.in1d(train_idxs, test_idxs).sum()
    check_3 = num_train_examples+len(val_idxs)+len(test_idxs) == num_imgs

    return train_idxs, val_idxs, test_idxs

def split_data(LOAD_IDXS, root_path, df_path, bee_data, num_imgs, test_split=0.2, val_split=0.1):
    df_path = os.path.join(root_path, f'orient_train_val_test_idxs.pt')

    if LOAD_IDXS:
        save_df_split_dict = torch.load(df_path)
        train_idxs = save_df_split_dict['train_idxs']
        val_idxs = save_df_split_dict['val_idxs']
        test_idxs = save_df_split_dict['test_idxs']
    else:
        train_idxs, val_idxs, test_idxs = train_val_test_split(bee_data, num_imgs, test_split, val_split)
        save_df_split_dict = {"train_idxs": train_idxs, "val_idxs": val_idxs, "test_idxs": test_idxs}
        # Save split indices to use later
        df_path = os.path.join(root_path, f'orient_train_val_test_idxs.pt')
        torch.save(save_df_split_dict, df_path)

    return train_idxs, val_idxs, test_idxs

class CustomLastLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Sigmoid()

    def forward(self, x):
        out = self.activation(x)
        out = out * 2.0 * np.pi
        return out

def build_resnet(num_classes):
    print(f"Building resnet-18 with {num_classes} class(es).")
    resnet = torchvision.models.resnet18(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                CustomLastLayer())
    return resnet

class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, pred_y, true_y):
        term_1 = torch.sin(true_y - pred_y)
        term_2 = torch.cos(true_y - pred_y)
        loss = (torch.atan2(term_1, term_2))**2
        return loss.mean()

def init_metrics():
    metrics = {
        "losses" : {
            "train" : [],
            "val"  : []
        },
        "accs" : {
            "train" : [],
            "val"  : []
        }
    }
    return metrics

def save_model(model, optimizer, metrics, epoch_i):
    save_dict = {
        'model'   : model.state_dict(),
        'optim'   : optimizer.state_dict(),
        'metrics' : metrics,
        'epoch'   : epoch_i
    }
    torch.save(save_dict, f'saved_models/ResnetOrientationModel_epoch{epoch_i:04d}.pt')

def get_accuracy(y, preds, tol, BATCH_SIZE):
    xxx = (y - tol/2) <= preds
    yyy = preds <= (y + tol/2)
    accuracy = (xxx.view(-1) & yyy.view(-1)).sum() / float(BATCH_SIZE) * 100
    return accuracy

def evaluate(test_loader, model, criterion, device, tol, BATCH_SIZE):
    batch_test_accs = []
    batch_test_loss = []
    with torch.no_grad():
        model.eval()
        for batch_i, (X, y, head_x, head_y, tail_x, tail_y) in enumerate(test_loader):
        # ----------------------
            X = Utils.convert_X_for_resnet(X)

            X = X.to(device)
            y = y.to(device)

            # Convert to float data type
            y = y.type(torch.FloatTensor).view(-1,1)
            y = np.deg2rad(y)  # Convert to radians
            y = y.to(device)

            logits = model(X)

            accuracy = get_accuracy(y, logits, tol, BATCH_SIZE)
            batch_test_accs.append(accuracy.item())

            loss = criterion(logits, y)
            batch_test_loss.append(loss.item())

            # stdout
            eval_stdout = f'\rBatch {batch_i+1}/{len(test_loader)} -- Loss: {np.mean(batch_test_loss):0.8f} -- Accuracy: {np.mean(batch_test_accs):0.8f}'
            sys.stdout.write(eval_stdout)
            sys.stdout.flush()

    mean_acc = sum(batch_test_accs) / len(batch_test_accs)
    mean_loss = sum(batch_test_loss) / len(batch_test_loss)
    model.train()
    return mean_acc, mean_loss

def training_loop(model, start_epoch, N_EPOCHS, train_loader, val_loader,
                  metrics, device, optimizer, criterion, SAVE_FREQ, tol, BATCH_SIZE):
    try:
        model.train()
        for epoch_i in range(start_epoch, start_epoch+N_EPOCHS):

            # Training loop
            # ------------------------------------------------
            print("Training:")
            batch_train_losses = []
            batch_train_accs = []
            for batch_i, (X_, y_, head_x, head_y, tail_x, tail_y) in enumerate(train_loader):
                X_, y_ = Utils.shuffle_batch(X_, y_)
                X_fixed = Utils.convert_X_for_resnet(X_)

                X_fixed = X_fixed.to(device)
                y_ = y_.to(device)

                # Convert to float data type
                y_ = y_.type(torch.FloatTensor).view(-1,1)
                y_ = np.deg2rad(y_)  # Convert to radians
                y_ = y_.to(device)

                # Pass through model
                # Logits as net's predictions
                logits = model(X_fixed)

                # Compute loss
                loss = criterion(logits, y_)
                batch_train_losses.append(loss.item())

                # Training step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Get training accs within tol% error
                with torch.no_grad():
                    xxx = (y_ - tol/2) <= logits
                    yyy = logits <= (y_ + tol/2)
                    accuracy = (xxx.view(-1) & yyy.view(-1)).sum() / float(BATCH_SIZE) * 100
                batch_train_accs.append(accuracy.item())

                print_str = f'\rEpoch {epoch_i+1}/{start_epoch+N_EPOCHS} -- Batch {batch_i}/{len(train_loader)}'
                print_str += f' -- Loss: {loss.item():0.8f} -- Train accuracy: {np.mean(batch_train_accs):0.4f}'
                sys.stdout.write(print_str)
                sys.stdout.flush()

            metrics['losses']['train'].append((epoch_i, np.mean(batch_train_losses)))
            metrics['accs']['train'].append((epoch_i, np.mean(batch_train_accs)))
            # ------------------------------------------------

            # Test Acc
            print("\nValidation:")
            with torch.no_grad():
                test_acc, test_loss = evaluate(val_loader, model, criterion, device, tol, BATCH_SIZE)
            metrics['accs']['val'].append((epoch_i, test_acc))
            metrics['losses']['val'].append((epoch_i, test_loss))
            print("\n")

            if epoch_i % SAVE_FREQ == 0:
                save_model(model, optimizer, metrics, epoch_i)

    except KeyboardInterrupt:
        save_model(model, optimizer, metrics, epoch_i)
        print("\nEnding early.")

    return model, metrics

def evaluate_plots(save_path, metrics, test_loader, train_loader, model, criterion, device, BATCH_SIZE):
    epochs = np.array(metrics['losses']['train'])[::,0]
    train_losses = np.array(metrics['losses']['train'])[::,1]
    test_losses = np.array(metrics['losses']['val'])[::,1]

    fig = plt.figure(figsize=(10,5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train');
    plt.plot(epochs[:len(test_losses)], test_losses, label='Val');
    plt.title(f'Loss curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    epochs = np.array(metrics['accs']['train'])[::,0]
    train_accs = np.array(metrics['accs']['train'])[::,1]
    test_accs = np.array(metrics['accs']['val'])[::,1]

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train');
    plt.plot(epochs[:len(test_losses)], test_accs, label='Val');
    plt.title(f'Accuracy curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(f'{save_path}/training_curves.png', dpi=150)

    # Compute accuracy with tolerances
    tolerance_array_deg = np.arange(0, 30+1.01, 5)
    tolerance_array_rad = np.deg2rad(tolerance_array_deg)

    accs_across_tolerances = []
    with torch.no_grad():
        model.eval()
        for tol_i, tol in enumerate(tolerance_array_rad):
            print(f'Tol: {np.ceil(np.rad2deg(tol))} or +- {np.rad2deg(tol)/2:0.2f}')
            test_acc, test_loss = evaluate(test_loader, model, criterion, device, tol, BATCH_SIZE)
            print('\n')
            accs_across_tolerances.append(test_acc)

    fig, ax = plt.subplots(dpi=100)
    ax.plot(tolerance_array_deg, accs_across_tolerances, marker='o', linewidth=1)
    # for acc_i, acc in enumerate(accs_across_tolerances):
    #     ax.annotate(f'{acc:0.2f}', ((tolerance_array_deg[acc_i])+0.4, accs_across_tolerances[acc_i]-4))
    plt.xlabel(f'Tolerance (degree)')
    plt.ylabel(f'Accuracy (%)')
    plt.title(f'Test accuracy across tolerance levels')
    plt.xticks(tolerance_array_deg)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.savefig(f'{save_path}/test_accurracy.png', dpi=150)

def setup_args():
    parser = argparse.ArgumentParser(description='Classify scenting bees!')
    parser.add_argument('-p', '--data_root', dest='data_root', type=str, default='../data/training_data/orientation_estimator')
    parser.add_argument('-l', '--load_idxs', dest='load_idxs', type=bool, default=False)
    parser.add_argument('-t', '--test_split', dest='test_split', type=float, default=0.2)
    parser.add_argument('-v', '--val_split', dest='val_split', type=float, default=0.1)
    parser.add_argument('-s', '--shuffle', dest='shuffle', type=bool, default=True)
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=8)
    parser.add_argument('-c', '--num_classes', dest='num_classes', type=int, default=1)
    parser.add_argument('-r', '--learning_rate', dest='learning_rate', type=float, default=0.000001)
    parser.add_argument('-m', '--load_model', dest='load_model', type=bool, default=False)
    parser.add_argument('-f', '--load_path', dest='load_path', type=str, default='saved_models/ResnetOrientationModel.pt')
    parser.add_argument('-e', '--num_epochs', dest='num_epochs', type=int, default=20)
    parser.add_argument('-q', '--save_freq', dest='save_freq', type=int, default=10)
    parser.add_argument('-w', '--deg_error', dest='deg_error', type=int, default=10)



    args = parser.parse_args()
    return args

def main():
    LOAD_IDXS = args.load_idxs
    TEST_SPLIT = args.test_split
    VAL_SPLIT = args.val_split
    SHUFFLE = args.shuffle
    BATCH_SIZE = args.batch_size
    NUM_CLASSES = args.num_classes
    LEARNING_RATE = args.learning_rate
    LOAD_MODEL = args.load_model
    LOAD_PATH = args.load_path
    N_EPOCHS = args.num_epochs
    SAVE_FREQ = args.save_freq

    # Degree of error to compute accuracy
    DEG_ERROR = args.deg_error
    tol = DEG_ERROR * np.pi/180

    print("\n---------- Training orientation estimator ----------")

    # ------------------------------------------------------------- #
    ##### DATASET & TRANSFORMS ######
    print('Setting up data handler and transforms...')
    root_path = args.data_root
    baseline_transforms = transforms.Compose([transforms.ToTensor()])
    bee_data = DataHandler.BeeDataset(root_path, baseline_transforms, augment_transforms=None, mode='train')
    num_imgs = len(bee_data.all_imgs)
    print(f'\nNumber of training images: {num_imgs}')

    # ------------------------------------------------------------- #
    ##### DATA SPLITTING & LOADER ######
    print('\nSplitting data into train, validation, test set...')
    df_path = os.path.join(root_path, f'orient_train_val_test_idxs.pt')
    train_idxs, val_idxs, test_idxs = split_data(LOAD_IDXS, root_path, df_path, bee_data, num_imgs, TEST_SPLIT, VAL_SPLIT)

    # ------------------------------------------------------------- #
    ##### SETTING UP DATA SAMPLERS ######
    print('\nSetting up data samplers...')
    sampler_train = SubsetRandomSampler(train_idxs)
    sampler_val = DataSamplers.SubsetIdentitySampler(indices=val_idxs)
    sampler_test = DataSamplers.SubsetIdentitySampler(indices=test_idxs)

    # ------------------------------------------------------------- #
    ##### SETTING UP DATA LOADERS ######
    print('\nSetting up data loaders...')
    train_loader = DataLoader(bee_data, batch_size=BATCH_SIZE, sampler=sampler_train, drop_last=True)
    val_loader = DataLoader(bee_data, batch_size=BATCH_SIZE, sampler=sampler_val, drop_last=True)
    test_loader = DataLoader(bee_data, batch_size=BATCH_SIZE, sampler=sampler_test, drop_last=True)

    # ------------------------------------------------------------- #
    ##### SET UP MODEL ######
    print('\nSetting up model...')
    model = build_resnet(NUM_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ------------------------------------------------------------- #
    ##### SET UP MODEL ######
    criterion = CustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ------------------------------------------------------------- #
    ##### CHECK LOADING MODEL ######
    if LOAD_MODEL:
        print(f"Loading model from {load_path}")
        load_dict = torch.load(load_path)
        model.load_state_dict(load_dict['model'])
        optimizer.load_state_dict(load_dict['optim'])
        metrics = load_dict['metrics']
        start_epoch = load_dict['epoch']
    else:
        print("Initializing new model.")
        metrics = init_metrics()
        start_epoch = 0

    # ------------------------------------------------------------- #
    ##### TRAIN ######
    print("\nTraining model...")
    model, metrics = training_loop(model, start_epoch, N_EPOCHS, train_loader, val_loader,
                                   metrics, device, optimizer, criterion, SAVE_FREQ, tol, BATCH_SIZE)

    # ------------------------------------------------------------- #
    ##### EVALUATE ######
    print('\nPlotting the training curves over time & evaluating model...')
    save_path = 'eval_visualization'
    evaluate_plots(save_path, metrics, test_loader, train_loader, model, criterion, device, BATCH_SIZE)

if __name__ == '__main__':
    args = setup_args()
    main()
