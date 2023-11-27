import cv2
import numpy as np
import utils.general as general_utils
from tqdm import tqdm


class VideoHandler:
    def __init__(self, vid_path, color=True, img_limit=None, img_skip=1, start_i=0, end_i=None):
        self.vid_path = vid_path
        self.img_limit = img_limit
        self.img_skip = img_skip
        self.start_i = start_i
        self.end_i = end_i

        self._open_stream(vid_path)
        self.num_images_loaded = 0

    def _open_stream(self, vid_path):
        self.cap = cv2.VideoCapture(vid_path)

    def __iter__(self):
        self.frame_i = 0
        self.num_images_loaded = 0
        return self

    def __next__(self):
        # Read frame and increment frame counter
        ret, frame = self.cap.read()
        self.frame_i += 1

        # Check for image limit
        condition_1 = self.img_limit and self.num_images_loaded >= self.img_limit
        condition_2 = self.end_i is not None and self.frame_i >= self.end_i
        if condition_1 or condition_2:
            raise StopIteration
        # Check image skip
        elif (self.frame_i % self.img_skip != 0) or (self.frame_i < self.start_i):
            frame = self.__next__()
        else:
            if frame is None:
                raise StopIteration

            self.num_images_loaded += 1

        return frame

def remove_lines(gray):
    rowvals = np.mean(np.sort(gray, axis=1)[:, -100:], axis=1)
    graymod = np.copy(gray).astype(float)
    graymod *= np.expand_dims(np.max(rowvals) / np.array(rowvals), axis=1)
    graymod = np.clip(graymod, 0, 255)
    return graymod.astype(np.uint32)

def remove_lines_BGR(img_BGR):
    img_BGR_no_lines = np.zeros_like(img_BGR)
    for c_i in range(3):
        img_BGR_no_lines[:, :, c_i] = remove_lines(img_BGR[:, :, c_i])

    return img_BGR_no_lines

def morphology_transform(img, transform_key):
    transform = morphology.__dict__[transform_key]
    img_transformed = transform(img)
    return img_transformed
from PIL import Image as im

def process_video(video_path):
    vid = VideoHandler(video_path)
    images = []
    print('Processing video...')
    with tqdm(total=int(vid.cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        for img_BGR in vid:
            overlay_img_src = remove_lines_BGR(img_BGR)
            images.append(overlay_img_src)
            pbar.update(1)
    frame = images[0]
    height, width = len(frame), len(frame[0])
    path = video_path.replace('.mp4', '_processed.mp4')
    video = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for image in images:
        video.write(image)
    video.release()
    print(f'Video exported to {path}')


print("Select root folder")
data_root = "data/processed"
src_processed_root = general_utils.select_file(data_root)
print("Select video to process")
video = general_utils.select_file(src_processed_root)
process_video(video)
