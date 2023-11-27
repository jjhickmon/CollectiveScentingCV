import cv2
import numpy as np
import utils.general as general_utils
from tqdm import tqdm

if __name__ == '__main__':
    print("select root folder...")
    src_processed_root = general_utils.select_file("data/processed")
    print("select video from list...")
    video = general_utils.select_file(src_processed_root)
    VIDEO_NAME = video.split("/")[-1].replace(".mp4", "")
    print(src_processed_root)
    print(video)

    c = cv2.VideoCapture(video)
    _, f = c.read()

    count = 0
    n = 1 # frame skip amount
    length = int(c.get(cv2.CAP_PROP_FRAME_COUNT) / n)

    frames = []
    with tqdm(total=length) as pbar:
        while (1):
            success, f = c.read()
            if not success:
                break

            frames.append(f)
            count += n
            c.set(cv2.CAP_PROP_POS_FRAMES, count)

            k = cv2.waitKey(20)
            if k == 27:
                break
            pbar.update(1)

    print("Generating background image...")
    median = np.median(frames, axis=0).astype(dtype=np.uint8)

    cv2.imwrite(f'{src_processed_root}/background_{VIDEO_NAME}.png', median)
    cv2.destroyAllWindows()
    c.release()
