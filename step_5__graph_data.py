import utils.general as general_utils
import matplotlib.pyplot as plt
import json
import math
import cv2
from tqdm import tqdm
import pandas
import numpy as np
from utils.settings import COLORS

def imgs2vid(imgs, outpath, fps):
    ''' Stitch together frame imgs to make a movie. '''
    height, width, layers = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(outpath, fourcc, fps, (width, height), True)

    for img_i, img in enumerate(imgs):
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

def get_correlation_coefficient(bee_time_series):
    print(bee_time_series.keys())
    phi_coefficient = np.corrcoef([bee_time_series[bee_name] for bee_name in bee_time_series.keys()])[0, 1] # simplified form
    print(f"Standard Phi coefficient: {phi_coefficient}")
    jaccard_coefficient = np.sum(np.logical_and(bee_time_series["crop_0000"], bee_time_series["crop_0001"])) / np.sum(np.logical_or(bee_time_series["crop_0000"], bee_time_series["crop_0001"]))
    print(f"Standard Jaccard coefficient: {jaccard_coefficient}")

    best_pos_offset = 0
    best_neg_offset = 0
    best_pos_phi_coefficient = -1
    best_pos_jaccard_coefficient = -1
    best_neg_phi_coefficient = -1
    best_neg_jaccard_coefficient = -1
    for offset in range(1, 30*10, 1):
        offset_phi_coefficient = np.corrcoef(bee_time_series["crop_0000"][offset:], bee_time_series["crop_0001"][:-offset])[0, 1]
        offset_jaccard_coefficient = np.sum(np.logical_and(bee_time_series["crop_0000"][offset:], bee_time_series["crop_0001"][:-offset])) / np.sum(np.logical_or(bee_time_series["crop_0000"][offset:], bee_time_series["crop_0001"][:-offset]))
        if offset_phi_coefficient > best_pos_phi_coefficient and offset_jaccard_coefficient > best_pos_jaccard_coefficient:
            best_pos_phi_coefficient = offset_phi_coefficient
            best_pos_jaccard_coefficient = offset_jaccard_coefficient
            best_pos_offset = offset

        offset_phi_coefficient = np.corrcoef(bee_time_series["crop_0000"][:-offset], bee_time_series["crop_0001"][offset:])[0, 1]
        offset_jaccard_coefficient = np.sum(np.logical_and(bee_time_series["crop_0000"][:-offset], bee_time_series["crop_0001"][offset:])) / np.sum(np.logical_or(bee_time_series["crop_0000"][:-offset], bee_time_series["crop_0001"][offset:]))
        if offset_phi_coefficient > best_neg_phi_coefficient and offset_jaccard_coefficient > best_neg_jaccard_coefficient:
            best_neg_phi_coefficient = offset_phi_coefficient
            best_neg_jaccard_coefficient = offset_jaccard_coefficient
            best_neg_offset = offset
    print(f"Best positive offset (worker 0 offset foward in time): {best_pos_offset/30.0} seconds or {best_pos_offset} frames")
    print(f"Best positive offset Phi coefficient: {best_pos_phi_coefficient}")
    print(f"Best positive offset Jaccard coefficient: {best_pos_jaccard_coefficient}")

    print(f"Best negative offset (worker 1 offset foward in time): {best_neg_offset/30.0} seconds or {best_neg_offset} frames")
    print(f"Best negative offset Phi coefficient: {best_neg_jaccard_coefficient}")
    print(f"Best negative offset Jaccard coefficient: {best_neg_jaccard_coefficient}")

def graph_dist_to_queen(data_log, labels):
    distances = []
    frames = []
    for frame, data in data_log.items():
        queen_position = None
        worker_positions = []
        for bee in data:
            if int(bee['label']) == labels['queen']:
                queen_position = (bee['x'], bee['y'])
        if queen_position is not None: # TODO: handle case where queen is not detected
            worker_distances = {}
            for bee in data:
                if int(bee['label']) in labels['workers']:
                    worker_distances[bee['label']] = math.dist(queen_position, (bee['x'], bee['y']))
            distances.append(worker_distances)
            frames.append(int(frame.split('_')[1]))

    for worker in labels['workers']:
        label = str(worker)
        worker_dist = [dist[label] if label in dist.keys() else None for dist in distances]
        plt.plot(frames, worker_dist, label=f'Worker {worker}')
    plt.legend()
    plt.show()

def graph_position_map(data_log, video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    prev_positions = []
    for frame_name, data in tqdm(data_log.items()):
        for bee in data:
            bee_position = (bee['x'], bee['y'])
            bee_label = bee['label']
            color = COLORS[int(bee_label) % len(COLORS)]
            frame_num = int(frame_name.split('_')[1])
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_num)
            ret, frame = cap.read()
            for prev_position in prev_positions:
                cv2.circle(frame, prev_position['position'], 3, prev_position['color'], -1)
            cv2.circle(frame, bee_position, 3, color, -1)
            frames.append(frame)
            prev_positions.append({'position':bee_position, 'color':color})
    print("Saving video...")
    save_path = video_path.replace('.mp4', '_positions.mp4')
    imgs2vid(frames, save_path, 30)

def graph_interval_data(data_log_scenting, test_name):
    bee_interval_data = {}
    num_bees = len(data_log_scenting.groupby('cropped_number'))
    for bee in data_log_scenting.groupby('cropped_number'):
        bee_name = bee[0]

        interval_frames = 0
        intervals = {}
        bee_data = data_log_scenting.groupby('cropped_number').get_group(bee_name)
        for frame_name, classification in zip(bee_data["frame_num"], bee_data["classification"]):
            if classification == "scenting":
                frame_num = int(frame_name.split('_')[1])
                intervals[frame_num] = interval_frames
                interval_frames = 0
            else:
                interval_frames += 1
        bee_interval_data[bee_name] = intervals

    fig, axes = plt.subplots(nrows=num_bees, ncols=1,
                           figsize=(num_bees*7,7))
    for bee_name, bee_intervals in bee_interval_data.items():
        time_seconds = [frame_num/30.0 for frame_num in bee_intervals.keys()]
        interval_seconds = [interval_frames/30.0 for interval_frames in bee_intervals.values()]
        bee_label = int(bee_name.split("_")[-1])
        axes[bee_label] = plt.subplot(num_bees, 1, bee_label+1)
        bee_color = [color / 255 for color in reversed(COLORS[(bee_label+1)%len(COLORS)])] # convert to 0-1 range from bgr to rgb

        # NOTE: Additional plotting options
        # plt.plot(time_seconds, interval_seconds, label=f'Worker {bee_label}', linewidth='.5')
        # plt.scatter(time_seconds, interval_seconds, label=f'Worker {bee_label}', c='#ff7f0e', s=2.5)
        # plt.bar(time_seconds, interval_seconds, width=(1/30), log=True, label=f'Worker {bee_label}')
        axes[bee_label] = plt.hist(interval_seconds, bins=np.arange(0, 7, 1/8), log=True,label=f'Worker {bee_label}', color=(max(0 ,bee_color[0]-.1), max(0, bee_color[1]-.1), max(0, bee_color[2]-.1))) # grouped by 1/8 seconds
        plt.legend()

    fig.supxlabel("interval length (seconds)")
    fig.supylabel("number of scenting events")
    plt.suptitle(f"Histogram of scenting intervals for {test_name}")
    plt.get_current_fig_manager().set_window_title("intervals")
    plt.show()

def graph_time_series(data_log_scenting, test_name):
    bee_time_series = {}
    num_bees = len(data_log_scenting.groupby('cropped_number'))
    for bee in data_log_scenting.groupby('cropped_number'):
        bee_name = bee[0]
        classification = bee[1]["classification"]
        bee_time_series[bee_name] = [1 if scenting == "scenting" else 0 for scenting in classification]


    fig, axes = plt.subplots(nrows=num_bees, ncols=1,
                           figsize=(num_bees*7,7))

    get_correlation_coefficient(bee_time_series)

    for bee_name, bee_classification in bee_time_series.items():
        bee_label = int(bee_name.split("_")[-1])
        time_seconds = [frame_num/30.0 for frame_num in range(len(bee_classification))]
        classification_downsampled = []
        min_interval = 5 # 1/6th of a second
        for i in range(len(bee_classification)):
            if bee_classification[i] == 0:
                if 1 in bee_classification[max(0, i - min_interval): i]: # if there is a scenting event in the last min_interval frames
                    classification_downsampled.append(1)
                else:
                    classification_downsampled.append(0)
            else:
                classification_downsampled.append(1)
        axes[bee_label] = plt.subplot(num_bees, 1, bee_label+1)
        bee_color = [color / 255 for color in reversed(COLORS[(bee_label+1)%len(COLORS)])] # convert to 0-1 range from bgr to rgb

        # NOTE: Additional plotting options
        # axes[bee_label].bar(time_seconds, classification_downsampled, width=(1/30.0), label=f'Worker {bee_label+1}', color=bee_color, linewidth=.1, edgecolor='black')
        # axes[bee_label].scatter(time_seconds, bee_classification, label=f'Worker {bee_label}', c=bee_color, s=.5)
        axes[bee_label].plot(time_seconds, bee_classification, label=f'Worker {bee_label}', linewidth='1', color=(max(0 ,bee_color[0]-.1), max(0, bee_color[1]-.1), max(0, bee_color[2]-.1)))
        axes[bee_label].set_yticks([0, 1])
        axes[bee_label].set_yticklabels(['not scenting', 'scenting'])
        plt.legend()
    fig.supxlabel('seconds')
    fig.supylabel('classification')

    plt.suptitle(f"Time series data of scenting classifications for {test_name}")
    plt.get_current_fig_manager().set_window_title("intervals")
    plt.show()


if __name__ == '__main__':
    print("-- Select folder from list...")
    src_processed_root = general_utils.select_file("data/processed")
    data_log = json.load(open(f"{src_processed_root}/data_log.json"))
    data_log_scenting = pandas.read_json(open(f"{src_processed_root}/data_log_scenting.json"))
    test_name = src_processed_root.split("/")[-1]

    graph = "graph_histogram"
    if graph == "graph_position_map":
        print("-- Select video to process from list...")
        video_path = general_utils.select_file(src_processed_root)

    # NOTE: increase font size for poster
    # plt.rcParams.update({'font.size': 20})

    # NOTE: graphing options
    # labels = {'queen': 0, 'workers': [1,3]}
    # graph_dist_to_queen(data_log, labels)
    # graph_position_map(data_log, video_path)
    # graph_interval_data(data_log_scenting, test_name)
    graph_time_series(data_log_scenting, test_name)