import cv2 as cv
import numpy as np
from moviepy.editor import *
import math
import tensorflow as tf

"""
Generate .npy files for input to the rgb_imagenet model. 
1. Split each video into 5 second subclips.
2. Sample subclip frames at 25 fps.
3. Resize videos preserving aspect ratio so that the smallest dimension is 256 pixels, with bilinear interpolation.
4. Pixel values rescaled between -1 and 1.
5. Write video as .npy file with shape (1, num_frames, 224, 224, 3).
"""

LABELS = {"l": 0, "m": 1, "h": 2}


def make_subclips(params):
    videos = [i for i in os.listdir(params.videos_path) if i.endswith(".mp4")]
    os.chdir(params.videos_path)
    for v in videos:
        clip_num = 0
        clip_start = 0
        clip_end = params.window_size
        full_vid = VideoFileClip(v)
        vid_end = full_vid.end
        while clip_end < vid_end:
            subclip = full_vid.subclip(clip_start, min(clip_end, vid_end))
            clip_start = clip_end
            clip_end += params.window_size
            subclip_path = os.path.join(params.data_path, v[:-4] + "_" + str(clip_num) + ".mp4")
            print(subclip_path)
            if not os.path.exists(subclip_path):
                subclip.write_videofile(subclip_path)
            clip_num += 1


def make_npy(is_training, params):
    img_npys = []
    IMG_SIZE = params.img_size
    FRAME_RATE = params.frame_rate
    make_subclips(params)
    os.chdir(params.data_path)
    if is_training:
        vids = [i for i in os.listdir(params.data_path) if i.endswith(".mp4") and i[:2]in params.train_ids]
    else:
        vids = [i for i in os.listdir(params.data_path) if i.endswith(".mp4") and i[:2] == params.val_id]
    for s in vids:
        print(s)
        cap = cv.VideoCapture(s)
        vid = VideoFileClip(s)
        length = vid.duration
        num_frames = int(math.floor(30 * length))
        del vid
        sample_at = num_frames/FRAME_RATE  # sample every few frames
        frame_no = 0
        npy_index = 0
        label = LABELS[s[3]]
        print(num_frames, label)
        video_frames = np.zeros((num_frames, IMG_SIZE, IMG_SIZE, 3))

        while frame_no < num_frames:
            if frame_no % sample_at == 0:
                ret, frame = cap.read()
                old_height, old_width = frame.shape[0], frame.shape[1]
                dim = (old_width * 256 // old_height, 256)
                frame = cv.resize(frame, dim, interpolation=cv.INTER_LINEAR)
                center = (frame.shape[0] // 2, frame.shape[1] // 2)
                x = center[1] - IMG_SIZE // 2
                y = center[0] - IMG_SIZE // 2
                frame = frame[y:y + IMG_SIZE, x:x + IMG_SIZE]
                frame = frame.astype('float32')
                frame = frame/255 * 2 - 1
                video_frames[npy_index] = frame
                npy_index += 1
            frame_no += 1
        img_npys.append(label)
        img_npys.append(video_frames)
    return img_npys


def train_test_split(is_training, params, data_dir):
    """
    1-7, 11 include subject face.
    Use 1-6 for training
    Use 7 for validation
    Use 11 for testing
    Can see if model works on blurred faces as well?
    """
    img_npy = make_npy(is_training, params)

    def make_labels(input_arr, dest_data, dest_labels):
        os.chdir(data_dir)
        for n in input_arr:
            label = n[0]
            data = n[1]
            print(label)
            data = np.ndarray.astype(data, "float32")
            dest_data.append(data)
            dest_labels.append(label)
        dest_data = np.array(dest_data)
        dest_labels = np.array(dest_labels)
        print(dest_labels)
        if is_training:
            params.train_size = len(dest_data)
            dataset = (tf.data.Dataset.from_tensor_slices((dest_data, dest_labels)).batch(params.batch_size).prefetch(1))
        else:
            dataset = (tf.data.Dataset.from_tensor_slices((dest_data, dest_labels)).batch(1).prefetch(1))
        return dataset

    def make_input(input_dataset):
        iterator = input_dataset.make_initializable_iterator()
        images, labels = iterator.get_next()
        iterator_init_op = iterator.initializer

        inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
        return inputs

    dataset = make_labels(img_npy, train_data, train_labels)

    inputs = make_input(dataset)
    return inputs
