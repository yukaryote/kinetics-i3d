import cv2 as cv
import os
import numpy as np
from moviepy.editor import *
import time
import math
import argparse
import tensorflow as tf
from math import ceil

"""
Generate .npy files for input to the rgb_imagenet model. 
1. Split each video into 5 second subclips.
2. Sample subclip frames at 25 fps.
3. Resize videos preserving aspect ratio so that the smallest dimension is 256 pixels, with bilinear interpolation.
4. Pixel values rescaled between -1 and 1.
5. Write video as .npy file with shape (1, num_frames, 224, 224, 3).
"""

parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='path to subclips')
args = parser.parse_args()

DATASET = args.data_path
LABELS = {"l": "low risk", "m": "medium risk", "h": "high risk"}
NPY_PATH = os.path.join(args.data_path, "npy")
SUBCLIPS_PATH = os.path.join(args.data_path, "subclips")

IMG_SIZE = 224
FRAME_RATE = 25


def make_subclips():
    os.chdir(SUBCLIPS_PATH)
    videos = [i for i in os.listdir(SUBCLIPS_PATH) if i.endswith(".mp4")]
    for v in videos:
        clip_num = 0
        clip_start = 0
        clip_end = 5
        full_vid = VideoFileClip(v)
        vid_end = full_vid.end
        while clip_end < vid_end:
            subclip = full_vid.subclip(clip_start, min(clip_end, vid_end))
            clip_start = clip_end
            clip_end += 5
            subclip_path = v[:-4] + "_" + str(clip_num) + ".mp4"
            print(subclip_path)
            if not os.path.exists(subclip_path):
                subclip.write_videofile(subclip_path)
            clip_num += 1


def make_npy():
    os.chdir(SUBCLIPS_PATH)
    for s in os.listdir(SUBCLIPS_PATH):
        if not os.path.isfile(s[:-4] + ".npy"):
            print(s)
            cap = cv.VideoCapture(s)
            vid = VideoFileClip(s)
            num_frames = math.ceil(FRAME_RATE * vid.end)
            frame_no = 0
            print(num_frames)
            final_input = np.zeros((1, num_frames, IMG_SIZE, IMG_SIZE, 3))

            while frame_no < num_frames:
                ret, frame = cap.read()
                old_height, old_width = frame.shape[0], frame.shape[1]
                dim = (old_width * 256 // old_height, 256)
                frame = cv.resize(frame, dim, interpolation=cv.INTER_LINEAR)
                center = (frame.shape[0] // 2, frame.shape[1] // 2)
                x = center[1] - IMG_SIZE // 2
                y = center[0] - IMG_SIZE // 2
                frame = frame[y:y + IMG_SIZE, x:x + IMG_SIZE]
                norm_pixels = np.zeros((3,))
                for i in range(IMG_SIZE):
                    for j in range(IMG_SIZE):
                        pixel = frame[i, j][:]
                        for k in range(len(pixel)):
                            norm = pixel[k] / 255 * 2 - 1
                            norm_pixels[k] = norm
                        final_input[0][frame_no][i][j] = norm_pixels
                frame_no += 1
            np.save(s[:-4], final_input)


def train_test_split(batch_size):
    npy_files = [i for i in os.listdir(DATASET) if i.endswith(".npy")]
    # Load the training data into two NumPy arrays, for example using `np.load()`.
    all_data = np.ndarray(shape=(len(npy_files),))
    all_labels = np.ndarray(shape=(len(npy_files),))
    for n in npy_files:
        label = LABELS[n[3]]
        data = np.load(n)
        np.append(all_data, data)
        np.append(all_labels, label)

    dataset = tf.data.Dataset.from_tensor_slices((all_data, all_labels))
    dataset = dataset.shuffle(len(npy_files))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    num_batches = ceil(len(npy_files)/batch_size)
    return dataset, num_batches
