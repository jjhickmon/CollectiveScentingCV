# Collective Olfactory Communication in Honey Bees
# Computer Vision Pipeline

## Overview:
This repo provides the code for the computer vision/deep learning pipeline used to analyze honey bee experimental data in [(Nguyen et al. 2020)](https://www.biorxiv.org/content/10.1101/2020.05.23.112540v1). The pipeline primarily includes dense object detection of individual bees in videos, classification of bees into scenting bees (wide wing angles as primary proxy for scenting), and estimation of the bees' body orientations.

## Main requirements (versions tested on):
- Python 3.8.3
- NumPy 1.17.4
- mH5py 2.10.0
- Matplotlib 3.1.1
The complete list of required packages provided in requirements.txt, which you can install in your environment with the command pip install -r requirements.txt. Setting up a Python virtual environment, such as [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), is highly recommended.

## Step 1. Detect bees

This is a semi-automatic process that uses Otsu’s adaptive thresholding, morphological transformations, and connected components to detect individual bees in the frames. Bees that are in clusters (i.e. touching or overlapping with one another) are detected as clusters of various sizes.

Input:
Images of frames extracted from video. Images should be stored in *data/processed/{folder_name}/{frames_folder_name}/*. A small dataset is provided here, and should be unzipped and placed in *data/processed/*. Inside the dataset folder, *denoised_frames/* holds the images from a short video. There is also a folder *UI_annotation_history* that holds sample data for the annotation desribed below.

Usage:
`python step_1__run_detection.py` takes in the (preprocessed) frame images and launches an interactive GUI with one frame (i.e. the first frame) and allows the user to click on centers of individual bees that are expected to be detected as individuals. Bees that touch or overlap one another should not be labeled and will be automatically detected as clusters. See example below for the GUI and bees that should be labeled (green dots). After the user finishes labeling this frame, the algorithm will use the labels to search for parameters that will maximize accuracy of the algorithm's predictions checked against the user-provided labels. The best parameters are then used to automatically process the rest of the frames to detect individual bees and bees in clusters.

Command line parameters:
- `-p` or `--data_root`: Path to the data folder (default: `data/processed`)
- `-r` or `--fps`: Frame per second for output movie (default: 25)
- `-v` or `--verbose`: FFMPEG Verbosity when visualizing (default: False)
- `-f` or `--force`: Force overwrite in data folder (default: True)
- `-c` or `--draw_clusters`: Draw cluster detections (default: True)
- `-t` or `--draw_trash`: Draw trash detections (default: False)
- `-u` or `--prevUI`: Use previous UI results (default: False)

Output:
A folder called *UI_annotation_history* will be made in the data folder to store the annotation history for future uses. After the algorithm detects bees, *data_log.json* will be created and it stores information of all the detected bees: the x, y positions of the bounding box top left corner and the width and height of that box. For visualization, a folder `detection_frames` and movie `detection_movie.mp4` will be created to show the output detections.

## Step 2. Classify scenting bees & estimate orientation
