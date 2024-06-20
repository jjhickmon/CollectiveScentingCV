import cv2
import numpy as np
import utils.general as general_utils
from tqdm import tqdm
import os

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


bee_color = []
def process_video(video_path, frames_path):
    # vid = VideoHandler(video_path)
    images = []
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    # add all images in raw frames path to list
    print('Processing video...')

    skip_frames = 100
    for filename in tqdm(sorted(os.listdir(frames_path))[::skip_frames]):
        img_BGR = cv2.imread(os.path.join(frames_path, filename))
        if img_BGR is None:
            print(f'Error: Could not read frame {filename}. Skipping...')
            continue
        images.append(img_BGR)

    def mouse_callback(event, x, y, flags, param):
        global bee_color
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get the BGR color at the clicked position
            bgr_color = images[0][y, x]

            # Print BGR and converted RGB color values
            print("BGR Color:", bgr_color)
            rgb_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2RGB)
            print("RGB Color:", rgb_color[0][0])
            bee_color = rgb_color[0][0]

    cv2.imshow("frame", images[0])
    cv2.setMouseCallback("frame", mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    bee_color_range = 100
    lower = np.array([bee_color[0] - bee_color_range, bee_color[1] - bee_color_range, bee_color[2] - bee_color_range])
    upper = np.array([bee_color[0] + bee_color_range, bee_color[1] + bee_color_range, bee_color[2] + bee_color_range])

    for i in range(len(images)):
        fg_mask = cv2.inRange(images[i], lower, upper)
        bg_mask = cv2.bitwise_not(fg_mask)
        images[i] = cv2.bitwise_and(images[i], images[i], mask=bg_mask)
        bg_subtractor.apply(images[i])
        cv2.imshow("frame", images[i])
        cv2.waitKey(0)

    print("bee color", bee_color)

    # print('Processing video...')
    # with tqdm(total=int(vid.cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
    #     for img_BGR in vid:
    #         overlay_img_src = remove_lines_BGR(img_BGR)
    #         images.append(overlay_img_src)
    #         pbar.update(1)

    frame = images[0]
    height, width = len(frame), len(frame[0])
    # set image width and height
    # for i in range(len(images)):
    #     if images[i].shape[0] != height or images[i].shape[1] != width:
    #         images[i] = cv2.resize(images[i], (width, height))
    print(width, height)
    # path = video_path.replace('.mp4', '_processed.mp4')
    # video = cv2.VideoWriter(
    #     path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))
    # for image in images:
    #     video.write(image)
    # video.release()
    # print(f'Video exported to {path}')

    # background = np.median(images, axis=0).astype(dtype=np.uint)
    background = bg_subtractor.getBackgroundImage()
    print("background acquired")
    cv2.imwrite(f'{src_processed_root}/background_q2bs_1.png', background)
    print(f'Background exported to {src_processed_root}/background_q2bs_1.png')

print("Select root folder")
data_root = "data/processed"
src_processed_root = general_utils.select_file(data_root)
print("Select video")
video_path = general_utils.select_file(src_processed_root)
print("Select denoised frames")
frames_path = general_utils.select_file(src_processed_root)
process_video(video_path, frames_path)
