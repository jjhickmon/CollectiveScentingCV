# Honeybee Swarm Dynamics: Investigating the Relationship Between Individual Decision Making and Collective Foraging
# Computer Vision Pipeline

## Overview:
This repo builds off of the code for the computer vision/deep learning pipeline used to analyze honey bee experimental data in [(Nguyen et al. 2020)](https://www.biorxiv.org/content/10.1101/2020.05.23.112540v1). The pipeline focuses on spatiotemporal consistency when tracking individual bees in videos, classification of bees into scenting bees (wide wing angles as primary proxy for scenting), and estimation of the bees' body orientations.

## Main requirements (versions tested on):
- Python 3.6
- NumPy 1.18.5
- OpenCV 4.1.1.26
- PyTorch 1.3.1
- PyTorch TorchVision 0.4.2
- Matplotlib 3.1.3
- [FFmpeg](https://ffmpeg.org/)

The complete list of required packages (besides FFmpeg) provided in *requirements.txt*, which you can install in your environment with the command `pip install -r requirements.txt`. Setting up a Python virtual environment, such as [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) with `pip`, is highly recommended.

## Note
The changes have not yet been tested on other devices. Email javonh@cs.washington.edu if there are any issues when using this pipeline.
Some of the requirements listed in requirements.txt may no longer be needed. Feel free to only install necessary libraries.

<!-- ----------------------------------------------------------------------- -->

## Step 0. Get background

Since the background of our video is stationary we utilize a standard method of video background subtraction by defining the background to be the median of each frame. By using this temporal median approach, we can approximate the background pixels and utilize that for background subtraction.

## Step 1. Run tracker

This is a semi-automatic process that uses manual thresholding, morphological transformations, and our custom algorithm to detect individual bees in the frames. Bees that are in clusters (i.e. touching or overlapping with one another) are separated into the individuals that compose the cluster.

### Input:
Images of frames extracted from video. Images should be stored in *data/processed/{folder_name}/{frames_folder_name}/*. A small dataset is provided [here](https://drive.google.com/drive/folders/1adOMmJc2hFB4eaDnGkpJkUybTysl3bRh?usp=sharing), and should be unzipped and placed in *data/processed/*. Inside the dataset folder, *denoised_frames/* holds the images from a short video. There is also a folder *UI_annotation_history* that holds sample data for the annotation described below.

### Usage:
**`python step_1__run_tracker.py`** takes in the a video and background image from a chosen data folder and starts running the tracker. The first frame of the video should have no overlapping bees (or the pipeline should be modified to run manual labelling first).

**Steps**
1. Select the root folder for the video to track.
2. Select the video within the root folder to analyze
3. Select the background image which should have been extracted from the video

Files should be structured as follows:
- data/processed/{root_folder_name}/
- data/processed/{root_folder_name}/{video_name}
- data/processed/{root_folder_name}/{background_image_name}

### Output:
After the algorithm detects bees, *data_log.json* will be created and it stores information of all the detected bees: the x, y positions of the bounding box top left corner and the width and height of that box. For visualization, a folder *detection_frames* and movie *{VIDEO_NAME}_contours.mp4.mp4* will be created to show the output detections.

<!-- ----------------------------------------------------------------------- -->

## Step 2. Classify scenting bees

After detections in step 1, the individual bees can then be classified into scenting and non-scenting bees. We trained a ResNet-18 model for this binary classification task, and provide the trained model you can download [here](https://drive.google.com/file/d/11nA6UtDye5NATWTOW54LrWoLcc2kuFRl/view?usp=sharing).

### Input:
The trained model (.pt file) should be placed in *scenting_classification/saved_models*. Input data to be processed should be in *data/processed/{folder_name}*: the *data_log.json* and the frame images (e.g. *denoised_frames*).

### Usage:
**`python step_2__run_scenting_classification.py`** runs the detection data through the model to classify individual bees into scenting or non-scenting. Running on a GPU is highly recommended for speed.

**Command line parameters:**
- `-p` or `--data_root`: Path to the data folder (default: `data/processed`)
- `-m` or `--model_file`: Name of trained model (default: `ResnetScentingModel.pt`)
- `-b` or `--batch_size`: Batch size (default: `32`)
- `-c` or `--num_classes`: Number of classes (default: `2`)

### Output:
In the data folder for this specific movie, *data_log_scenting.json* will be created from this step. This builds upon *data_log.json* and adds a 'classification' to each bee in each frame.

### To retrain model:
Training data should be in *data/training_data/scenting_classifier*. Sample labeled data is provided [here](https://drive.google.com/drive/folders/14bVvOCAwD4TbqOD0GoJ1msMFzvYZIICg?usp=sharing).

Navigate to *scenting_classification* and run **`python train_scenting_classifier.py`**. Trained models will be stored in *saved_models*. At the end of training, evaluation plots (training curves, ROCs, confusion matrices) will be made and stored in *eval_visualization*.

**Command line parameters:**
- `-p` or `--data_root`: Path to the data folder (default: `'../data/training_data/scenting_classifier'`)
- `-l` or `--load_idxs`: Load previous data splitting indices (default: False)
- `-d` or `--dropout`: Apply drop out (default: 0.0)
- `-a` or `--augmentation`: Apply augmentations (default: True)
- `-t` or `--test_split`: Test set split proportion (default: 0.2)
- `-v` or `--val_split`: Validation set split proportion (default: 0.1)
- `-s` or `--shuffle`: Shuffle data sets (default: True)
- `-b` or `--batch_size`: Batch size (default: 32)
- `-c` or `--num_classes`: Number of classes (default: 2)
- `-r` or `--learning_rate`: Learning rate (default: 0.0001)
- `-m` or `--load_model`: Load a model (default: False)
- `-f` or `--load_path`: Path to model to load (default: `saved_models/ResnetScentingModel.pt`)
- `-e` or `--num_epochs`: Number of training epochs (default: 20)
- `-q` or `--save_freq`: Frequency for saving model (default: 10)

<!-- ----------------------------------------------------------------------- -->

## Step 3. Estimate orientations

We can also obtain the body orientation of the scenting bees to know their scenting directions. Another ResNet-18 model is trained for the regression task of estimating the orientation angle (head to tail) of bees. The trained model is provided [here](https://drive.google.com/open?id=11t5OYkj43LwlPKBGpeGw4E4OeKpwfopD).

### Input:
The trained model (.pt file) should be placed in *orientation_estimation/saved_models*. Input data to be processed should be in *data/processed/{folder_name}*: the *data_log_scenting.json* and the frame images (e.g. *denoised_frames*).

### Usage:
**`python step_3__run_orientation_estimator.py`** runs the classification data through the model to estimate the body orientation angle of individual bees. Running on a GPU is highly recommended for speed.

**Command line parameters:**
- `-p` or `--data_root`: Path to the data folder (default: `data/processed`)
- `-m` or `--model_file`: Name of trained model (default: `ResnetOrientationModel.pt`)
- `-b` or `--batch_size`: Batch size (default: `32`)
- `-c` or `--num_classes`: Number of classes (default: `1`)

### Output:
In the data folder for this specific movie, *data_log_orientation.json* will be created from this step. This builds upon *data_log.json* and *data_log_orientation.json* and adds an 'orientation' angle to each bee in each frame.

### To retrain model:
Training data should be in *data/training_data/orientation_estimation*. Sample labeled data is provided [here](https://drive.google.com/drive/folders/1f-nKt3Cy5w9SyTvO-l4fMPOKoFKRSB2e?usp=sharing).

Navigate to *orientation_estimation* and run **`python train_orientation_estimator.py`**. Trained models will be stored in *saved_models*. At the end of training, evaluation plots (training curves, degree tolerance) will be made and stored in *eval_visualization*.

**Command line parameters:**
- `-p` or `--data_root`: Path to the data folder (default: `'../data/training_data/scenting_classifier'`)
- `-l` or `--load_idxs`: Load previous data splitting indices (default: False)
- `-t` or `--test_split`: Test set split proportion (default: 0.2)
- `-v` or `--val_split`: Validation set split proportion (default: 0.1)
- `-s` or `--shuffle`: Shuffle data sets (default: True)
- `-b` or `--batch_size`: Batch size (default: 32)
- `-c` or `--num_classes`: Number of classes (default: 2)
- `-r` or `--learning_rate`: Learning rate (default: 0.0001)
- `-m` or `--load_model`: Load a model (default: False)
- `-f` or `--load_path`: Path to model to load (default: `saved_models/ResnetScentingModel.pt`)
- `-e` or `--num_epochs`: Number of training epochs (default: 20)
- `-q` or `--save_freq`: Frequency for saving model (default: 10)
- `-w` or `--deg_error`: Degree of error tolerated for evaluating during training (default: 10)

<!-- ----------------------------------------------------------------------- -->

## Step 4. Visualize scenting recognition data

After the whole detection and scenting recognition pipeline, we can make a movie of the output data to visualize the scenting bees and their scenting directions.

### Input:
The *data_log_orientation.json* from step 3 and the frame images (e.g. *denoised_frames*).

### Usage:
**`python step_4__visualize.py`** plots orientation arrows on the scenting bees and outputs a movie of all the frames provided.

**Command line parameters:**
- `-p` or `--data_root`: Path to the data folder (default: `data/processed`)
- `-r` or `--fps`: Frames per second of output movie (default: `15`)

### Output:
In the data folder for this specific movie, a folder *output_frames* will be created to store the annotated frames and the *output_movie.mp4* will be created to make a movie of all the annotated frames.

<!-- ----------------------------------------------------------------------- -->

## Step 5. Graph data (optional)

Optionally you can graph the data generated from the tracker using matplotlib.

### Input:
The *data_log.json* and *data_log_scenting.json* are used to generate the graphs. Make sure to select the root directory for the video.

### Usage:
**`python step_5__graph_data.py`** plots a graph of the data generated.

### Output:
In the data folder for this specific movie, a graph image will be created.

Reference:
Nguyen DMT, Iuzzolino ML, Mankel A, Bozek K, Stephens GJ, Peleg O (2020). Flow-Mediated Collective Olfactory
Communication in Honeybee Swarms. bioRxiv 2020.05.23.112540; doi: https://doi.org/10.1101/2020.05.23.112540.
