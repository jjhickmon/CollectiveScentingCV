###### IMPORTS ######
# General
import os
import sys
import cv2
import glob
import json
import shutil
import pickle
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
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

def split_data(LOAD_IDXS, root_path, df_path, img_paths, labels_list, test_split=0.2, val_split=0.1):
    if LOAD_IDXS:
        save_df_split_dict = torch.load(df_path)
        train_idxs = save_df_split_dict['train_idxs']
        val_idxs = save_df_split_dict['val_idxs']
        test_idxs = save_df_split_dict['test_idxs']
    else:
        train_idxs, val_idxs, test_idxs = Utils.train_val_test_split(img_paths, labels_list, test_split, val_split)
        save_df_split_dict = {"train_idxs": train_idxs, "val_idxs": val_idxs, "test_idxs": test_idxs}
        # Save split indices
        df_path = os.path.join(root_path, f'train_val_test_idxs.pt')
        torch.save(save_df_split_dict, df_path)
    return train_idxs, val_idxs, test_idxs

def apply_augmentations(AUGMENTATION):
    if AUGMENTATION:
        augment_transforms = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.ColorJitter(brightness=0.10),
                                transforms.RandomAffine(degrees=360, translate=(0.05,0.05), scale=(0.93,1.03), fillcolor=210)])
    else:
        augment_transforms = None
    return augment_transforms

def build_resnet(num_classes, dropout=0.0):
    print(f"\nBuilding resnet-18 with {num_classes} classes.")
    resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_ftrs, num_classes))
    return resnet

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

def training_loop(model, start_epoch, N_EPOCHS, train_loader, val_loader, metrics, device, optimizer, criterion, SAVE_FREQ):
    try:
        model.train()
        for epoch_i in range(start_epoch, start_epoch+N_EPOCHS):
            print("Train:")
            batch_train_losses = []
            batch_train_accs = []
            for batch_i, (X_, y_) in enumerate(train_loader):
                X_, y_ = Utils.shuffle_batch(X_, y_)
                X_fixed = Utils.convert_X_for_resnet(X_)

                X_fixed = X_fixed.to(device)
                y_ = y_.to(device)

                # Pass through model
                logits = model(X_fixed)

                # Compute loss
                loss = criterion(logits, y_)
                batch_train_losses.append(loss.item())

                # Training step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Get training accs
                with torch.no_grad():
                    preds = Utils.get_prediction(logits)
                    accuracy = Utils.get_accuracy(y_, preds)
                batch_train_accs.append(accuracy.item())

                sys.stdout.write(f'\rEpoch {epoch_i+1}/{start_epoch+N_EPOCHS} -- Batch {batch_i+1}/{len(train_loader)} -- Loss: {loss.item():0.5f} -- Train accuracy: {np.mean(batch_train_accs):0.5f}%')
                sys.stdout.flush()

            metrics['losses']['train'].append((epoch_i, np.mean(batch_train_losses)))
            metrics['accs']['train'].append((epoch_i, np.mean(batch_train_accs)))
            # ------------------------------------------------
            # Validation Acc
            print("\nValidation:")
            with torch.no_grad():
                val_acc, val_loss = Utils.evaluate(val_loader, model, criterion, device)
            metrics['accs']['val'].append(val_acc)
            metrics['losses']['val'].append(val_loss)

            if epoch_i % SAVE_FREQ == 0:
                save_model(model, optimizer, metrics, epoch_i)
            print("\n")
    except KeyboardInterrupt:
        print("\nEnding early.")

    print('Done training.')

    # Save model at end of training
    save_model(model, optimizer, metrics, epoch_i)

    return model, metrics

def save_model(model, optimizer, metrics, epoch_i):
    save_dict = {
        'model'   : model.state_dict(),
        'optim'   : optimizer.state_dict(),
        'metrics' : metrics,
        'epoch'   : epoch_i
    }
    torch.save(save_dict, f'saved_models/ResnetScentingModel_epoch{epoch_i:05d}.pt')

def evaluate_plots(save_path, metrics, test_loader, train_loader, model, criterion, device):
    # Plot training loss and accuracies over time
    epochs = np.array(metrics['losses']['train'])[::,0]
    train_losses = np.array(metrics['losses']['train'])[::,1]
    val_losses = np.array(metrics['losses']['val'])[:]

    fig = plt.figure(figsize=(10,5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train')
    plt.plot(epochs, val_losses, label='Validation')
    plt.title(f'Loss curves')
    plt.xlabel(f'Epoch')
    plt.ylabel(f'Loss')
    plt.legend()

    epochs = np.array(metrics['accs']['train'])[::,0]
    train_accs = np.array(metrics['accs']['train'])[::,1]
    val_accs = np.array(metrics['accs']['val'])[:]

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train')
    plt.plot(epochs, val_accs, label='Validation')
    plt.title(f'Accuracy curves')
    plt.xlabel(f'Epoch')
    plt.ylabel(f'Accuracy (%)')
    plt.legend()
    plt.savefig(f'{save_path}/training_curves.png', dpi=150)

    # Test accuracy, ROC, AUC, confusion matrix, f1-score
    test_acc, test_loss = Utils.evaluate(test_loader, model, criterion, device, verbose=False)
    print(f'\nTest accurracy: {test_acc:0.5f}%')

    # ROC
    print('\nComputing and plotting the ROCs...')
    thresholds = np.linspace(0, 1.0, 11)
    model = model.to('cpu')
    test_truth_outputs = Evaluation.evaluation_dataframe(thresholds, test_loader, model)
    test_truth_outputs.reindex(['tp', 'tn', 'fp', 'fn', 'tpr', 'fpr', 'predicted_ys', 'true_ys'])
    # test_truth_outputs.to_pickle('test_truth_outputs.pkl')
    Evaluation.plot_ROC(test_truth_outputs, 'test', 'orange', save_path)

    train_truth_outputs = Evaluation.evaluation_dataframe(thresholds, train_loader, model)
    train_truth_outputs.reindex(['tp', 'tn', 'fp', 'fn', 'tpr', 'fpr', 'predicted_ys', 'true_ys'])
    # train_truth_outputs.to_pickle('train_truth_outputs.pkl')
    Evaluation.plot_ROC(train_truth_outputs, 'train', 'g', save_path)

    # Confusion matrix, f1-score
    print('\nPlotting the confusion matrices...\n')
    Evaluation.plot_confusion_matrix(train_truth_outputs, 'train', 'Greens', save_path)
    Evaluation.plot_confusion_matrix(test_truth_outputs, 'test', 'Oranges', save_path)

def setup_args():
    parser = argparse.ArgumentParser(description='Classify scenting bees!')
    parser.add_argument('-p', '--data_root', dest='data_root', type=str, default='../data/training_data/scenting_classifier')
    parser.add_argument('-l', '--load_idxs', dest='load_idxs', type=bool, default=False)
    parser.add_argument('-d', '--dropout', dest='dropout', type=float, default=0.0)
    parser.add_argument('-a', '--augmentation', dest='augmentation', type=bool, default=True)
    parser.add_argument('-t', '--test_split', dest='test_split', type=float, default=0.2)
    parser.add_argument('-v', '--val_split', dest='val_split', type=float, default=0.1)
    parser.add_argument('-s', '--shuffle', dest='shuffle', type=bool, default=True)
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('-c', '--num_classes', dest='num_classes', type=int, default=2)
    parser.add_argument('-r', '--learning_rate', dest='learning_rate', type=float, default=0.0001)
    parser.add_argument('-m', '--load_model', dest='load_model', type=bool, default=False)
    parser.add_argument('-f', '--load_path', dest='load_path', type=str, default='saved_models/ResnetScentingModel.pt')
    parser.add_argument('-e', '--num_epochs', dest='num_epochs', type=int, default=20)
    parser.add_argument('-q', '--save_freq', dest='save_freq', type=int, default=10)
    args = parser.parse_args()
    return args

def main(args):
    LOAD_IDXS = args.load_idxs
    DROPOUT = args.dropout
    AUGMENTATION = args.augmentation
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

    print("\n---------- Training scenting classifier ----------")
    dir = 'eval_visualization'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

    # ------------------------------------------------------------- #
    ##### DATASET & TRANSFORMS ######
    print('Setting up data handler and transforms...')
    root_path = args.data_root
    baseline_transforms = transforms.Compose([transforms.ToTensor()])
    bee_data = DataHandler.BeeDataset(root_path, baseline_transforms, augment_transforms=None, mode='train')
    num_imgs = len(bee_data.all_img_paths)
    print(f'\nNumber of training images: {num_imgs}')

    # ------------------------------------------------------------- #
    ##### DATA SPLITTING & LOADER ######
    print('\nSplitting data into train, validation, test set...')

    df_path = os.path.join(root_path, f'train_val_test_idxs.pt')
    img_paths = bee_data.all_img_paths
    labels_list = bee_data.all_labels

    train_idxs, val_idxs, test_idxs = split_data(LOAD_IDXS, root_path, df_path, img_paths, labels_list, test_split=TEST_SPLIT, val_split=VAL_SPLIT)

    # ------------------------------------------------------------- #
    ##### APPLY TRANSFORMS TO TRAIN SET ######
    print('\nSetting up augmentation transforms for training data...')
    baseline_transforms = transforms.Compose([transforms.ToTensor()])

    augment_transforms = apply_augmentations(AUGMENTATION)

    bee_data = DataHandler.BeeDataset(root_path, baseline_transforms,
                                    augment_transforms=augment_transforms,
                                    mode='train', train_idxs=train_idxs)

    # ------------------------------------------------------------- #
    ##### SETTING UP DATA SAMPLERS ######
    print('\nSetting up data samplers...')
    balanced_sampler_train = DataSamplers.BalancedSubsetSampler_2(img_paths=img_paths, labels_list=labels_list, idxs=train_idxs)
    sampler_val = DataSamplers.SubsetIdentitySampler(indices=val_idxs)
    sampler_test = DataSamplers.SubsetIdentitySampler(indices=test_idxs)

    # ------------------------------------------------------------- #
    ##### SETTING UP DATA LOADERS ######
    print('\nSetting up data loaders...')
    train_loader = DataLoader(bee_data, batch_size=BATCH_SIZE, sampler=balanced_sampler_train, drop_last=True, num_workers=2)
    val_loader = DataLoader(bee_data, batch_size=BATCH_SIZE, sampler=sampler_val, drop_last=True)
    test_loader = DataLoader(bee_data, batch_size=BATCH_SIZE, sampler=sampler_test, drop_last=True)

    # ------------------------------------------------------------- #
    ##### SET UP MODEL ######
    print('\nSetting up augmentation transforms for training data...')
    model = build_resnet(NUM_CLASSES, DROPOUT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ------------------------------------------------------------- #
    ##### SET UP MODEL ######
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ------------------------------------------------------------- #
    ##### CHECK LOADING MODEL ######
    if LOAD_MODEL:
        print(f"Loading model from {LOAD_PATH}")
        load_dict = torch.load(LOAD_PATH)
        model.load_state_dict(load_dict['model'])
        optimizer.load_state_dict(load_dict['optim'])
        metrics = load_dict['metrics']
        start_epoch = load_dict['epoch']
    else:
        print("\nInitializing new model...")
        metrics = init_metrics()
        start_epoch = 0

    # ------------------------------------------------------------- #
    ##### TRAIN ######
    print("\nTraining model...")
    model, metrics = training_loop(model, start_epoch, N_EPOCHS, train_loader, val_loader,
                                   metrics, device, optimizer, criterion, SAVE_FREQ)

    # ------------------------------------------------------------- #
    ##### EVALUATE ######
    print('\nPlotting the training curves over time...')
    save_path = 'eval_visualization'
    evaluate_plots(save_path, metrics, test_loader, train_loader, model, criterion, device)

if __name__ == '__main__':
    args = setup_args()
    main(args)
