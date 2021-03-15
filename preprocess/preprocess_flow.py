import cv2 as cv
import os
import numpy as np
from moviepy.editor import *
import time
import math
import argparse

"""
Generate .npy files for input to the flow_imagenet model. 
1. Split each video into 5 second subclips.
2. Sample subclip frames at 25 fps.
3. Resize videos preserving aspect ratio so that the smallest dimension is 256 pixels, with bilinear interpolation.
4. Convert image to grayscale.
5. Apply TV-L1 optical flow
4. Truncate pixel values to between -20 and 20
4. Pixel values rescaled between -1 and 1.
5. Write video as .npy file with shape (1, num_frames, 224, 224, 3).
"""

parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='path to subclips')
args = parser.parse_args()

DATASET = args.data_path

IMG_SIZE = 224
FRAME_RATE = 25


def make_npy():
    os.chdir(DATASET)
    for s in os.listdir(DATASET):
        if not os.path.isfile(s[:-4] + ".npy"):
            print(s)
            cap = cv.VideoCapture(s)
            vid = VideoFileClip(s)
            num_frames = math.ceil(FRAME_RATE * vid.end)
            frame_no = 0
            print(num_frames)
            ret, frame_0 = cap.read()
            prvs = cv.cvtColor(frame_0, cv.COLOR_BGR2GRAY)
            final_input = np.zeros((1, num_frames, IMG_SIZE, IMG_SIZE, 2))

            while frame_no < num_frames:
                ret, frame_1 = cap.read()
                old_height, old_width = frame.shape[0], frame.shape[1]
                dim = (old_width * 256//old_height, 256)
                frame = cv.resize(frame, dim, interpolation=cv.INTER_LINEAR)
                center = (frame.shape[0] // 2, frame.shape[1] // 2)
                x = center[1] - IMG_SIZE // 2
                y = center[0] - IMG_SIZE // 2
                frame = frame[y:y+IMG_SIZE, x:x+IMG_SIZE]
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                dtvl1 = cv.cuda()
                flowDTVL1 = dtvl1.calc(frame_0, frame_1, None)
                norm_pixels = np.zeros((2,))
                for i in range(IMG_SIZE):
                    for j in range(IMG_SIZE):
                        pixel = frame[i, j][:]
                        for k in range(len(pixel)):
                            norm = pixel[k]/255 * 2 - 1
                            norm_pixels[k] = norm
                        final_input[0][frame_no][i][j] = norm_pixels
                frame_no += 1
            np.save(s[:-4], final_input)


if __name__ == "__main__":
    make_npy()
