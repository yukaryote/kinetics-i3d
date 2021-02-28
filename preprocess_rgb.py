import cv2 as cv
import os
import numpy as np
from moviepy.editor import *
import time
import math

"""
Generate .npy files for input to the rgb_imagenet model. 
1. Split each video into 5 second subclips.
2. Sample subclip frames at 25 fps.
3. Resize videos preserving aspect ratio so that the smallest dimension is 256 pixels, with bilinear interpolation.
4. Pixel values rescaled between -1 and 1.
5. Write video as .npy file with shape (1, num_frames, 224, 224, 3).
"""

DATASET = os.path.join("/Users/isabellayu/sr-dataset/videos")
SUBCLIPS = os.path.join(DATASET, "subclips")
os.chdir(DATASET)
videos = [i for i in os.listdir(DATASET) if i.endswith(".mp4")]

IMG_SIZE = 224
FRAME_RATE = 25


def make_subclips():
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
            subclip_path = "subclips/" + v[:-4] + "_" + str(clip_num) + ".mp4"
            print(subclip_path)
            if not os.path.exists(subclip_path):
                subclip.write_videofile(subclip_path)
            clip_num += 1


def make_npy():
    os.chdir(SUBCLIPS)
    for s in os.listdir(SUBCLIPS):
        print(s)
        cap = cv.VideoCapture(s)
        vid = VideoFileClip(s)
        num_frames = math.ceil(FRAME_RATE * vid.end)
        frame_no = 0
        final_input = np.zeros((1, num_frames, IMG_SIZE, IMG_SIZE, 3))

        while frame_no < num_frames:
            ret, frame = cap.read()
            cv.imshow('fr', frame)
            old_height, old_width = frame.shape[0], frame.shape[1]
            dim = (old_width * 256//old_height, 256)
            frame = cv.resize(frame, dim, interpolation=cv.INTER_LINEAR)
            center = (frame.shape[0] // 2, frame.shape[1] // 2)
            x = center[1] - IMG_SIZE // 2
            y = center[0] - IMG_SIZE // 2
            frame = frame[y:y+IMG_SIZE, x:x+IMG_SIZE]
            cv.imshow("fr", frame)
            norm_pixels = np.zeros((3,))
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
    # make_subclips()
    make_npy()
