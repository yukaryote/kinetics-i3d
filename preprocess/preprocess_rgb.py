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
    os.chdir(params.data_path)
    videos = [i for i in os.listdir(params.data_path) if i.endswith(".mp4")]
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


def make_npy(params):
    IMG_SIZE = params.img_size
    FRAME_RATE = params.frame_rate
    os.chdir(params.data_path)
    for s in os.listdir(params.data_path):
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


def train_test_split(params, data_dir):
    """
    1-7, 11 include subject face.
    Use 1-6 for training
    Use 7 for validation
    Use 11 for testing
    Can see if model works on blurred faces as well?
    """
    npy_files = [i for i in os.listdir(data_dir) if i.endswith(".npy")]
    train_files = [i for i in npy_files if i[:2] in params.train_ids]
    validation_files = [i for i in npy_files if i[:2] == params.eval_id]
    test_files = [i for i in npy_files if i[:2] == params.test_id]
    print(train_files)

    params.train_size = len(train_files)
    params.eval_size = len(validation_files)

    # Load the training data into two NumPy arrays, for example using `np.load()`.
    train_data = []
    validation_data = []
    test_data = []

    train_labels = []
    validation_labels = []
    test_labels = []

    def make_labels(input_files, dest_data, dest_labels):
        os.chdir(data_dir)
        for n in input_files:
            label = LABELS[n[3]]
            data = np.load(n)
            data = np.squeeze(data, axis=0)
            data = data.astype("float32")
            dest_data.append(data)
            dest_labels.append(label)
        dest_data = np.array(dest_data)
        dest_labels = np.array(dest_labels)
        print(dest_labels)

        dataset = (tf.data.Dataset.from_tensor_slices((dest_data, dest_labels)).batch(params.batch_size).prefetch(1))
        return dataset

    def make_input(input_dataset):
        iterator = input_dataset.make_initializable_iterator()
        images, labels = iterator.get_next()
        iterator_init_op = iterator.initializer

        inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
        return inputs

    train_dataset = make_labels(train_files, train_data, train_labels)
    validation_dataset = make_labels(validation_files, validation_data, validation_labels)
    test_dataset = make_labels(test_files, test_data, test_labels)

    train_inputs = make_input(train_dataset)
    validation_inputs = make_input(validation_dataset)
    test_inputs = make_input(test_dataset)
    return train_inputs, validation_inputs, test_inputs