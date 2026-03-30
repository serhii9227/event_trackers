import os
import sys
from os.path import join

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import argparse


os.sep = '/'
np.set_printoptions(suppress=True)


def stack_event(stack_name, index, root, stack_amount_1c2c, stack_amount_3c, decay_rate_3c):
    """Process events from CSV and save as stacked images."""

    seq_name = root.split('/')[-1]

    # Get frame count from img folder
    img_path = os.path.join(root, 'img').replace('\\', '/')
    frame_num = len(os.listdir(img_path))

    # Read events from CSV
    csv_path = os.path.join(root, 'events.csv').replace('\\', '/')
    df = pd.read_csv(csv_path)
    events = df[['t', 'x', 'y', 'p']].values

    # Get image shape from first frame
    first_img = cv2.imread(os.path.join(img_path, sorted(os.listdir(img_path))[0]))
    h, w = first_img.shape[:2]
    pic_shape = (h, w)

    # Generate evenly spaced frame timestamps
    t_min = int(events[0, 0])
    t_max = int(events[-1, 0])
    time_series = list(np.linspace(t_min, t_max, frame_num + 1, dtype=np.int64))

    # Filter events by time range
    events = events[events[:, 0] >= time_series[0]]
    events = events[events[:, 0] < time_series[-1]]

    # Create output folder
    stack_path = os.path.join(root, stack_name).replace('\\', '/')
    if not os.path.exists(stack_path):
        os.mkdir(stack_path)

    deal_event(index, events, time_series, pic_shape, stack_path, stack_amount_1c2c, stack_amount_3c, decay_rate_3c)


def process_event(pos_img, neg_img, null_img, event, pic_shape, stack_amount_1c2c):
    """Apply a single event to the image buffers."""
    x, y, p = int(event[1]), int(event[2]), int(event[3])
    if 0 < x < pic_shape[1] and 0 < y < pic_shape[0]:
        if p == 1:
            pos_img[y][x] = min(255, pos_img[y][x] + stack_amount_1c2c)
        else:
            neg_img[y][x] = min(255, neg_img[y][x] + stack_amount_1c2c)


def save_2C_img(pos_img, neg_img, null_img, root):
    """Save 3-channel image (pos, neg, hidden)."""
    two_channel_img = np.zeros((pos_img.shape[0], pos_img.shape[1], 3), dtype=np.uint8)
    two_channel_img[:, :, 0] = pos_img
    two_channel_img[:, :, 1] = neg_img
    two_channel_img[:, :, 2] = null_img
    cv2.imwrite(root, two_channel_img)


def hidden_pic_generator(pos_img, neg_img, last_pos_pic, last_neg_pic, hidden_pic, stack_amount_3c, decay_rate_3c):
    """Generate new hidden state image."""
    new_hidden_pic = hidden_pic * decay_rate_3c
    pos_condition = (last_pos_pic == 0) & (pos_img != 0)
    neg_condition = (last_neg_pic == 0) & (neg_img != 0)
    new_hidden_pic[pos_condition] += stack_amount_3c
    new_hidden_pic[neg_condition] += stack_amount_3c
    new_hidden_pic = np.clip(new_hidden_pic, 0, 255)
    return new_hidden_pic


def deal_event(index, events, frame_timestamp, pic_shape, save_name, stack_amount_1c2c, stack_amount_3c, decay_rate_3c):
    """Split events into per-frame images and save."""

    flag = False
    last_pos_pic = np.full(pic_shape, 0, dtype=np.uint8)
    last_neg_pic = np.full(pic_shape, 0, dtype=np.uint8)
    hidden_state = np.full(pic_shape, 0, dtype=np.uint8)

    i = 1
    pos_img = np.full(pic_shape, 0, dtype=np.uint8)
    neg_img = np.full(pic_shape, 0, dtype=np.uint8)
    sub_index = 1
    T_num = 2
    sub_frame = np.linspace(frame_timestamp[0], frame_timestamp[1], T_num)

    for event in tqdm(events, desc="{} Writing {} events".format(index, save_name.split('/')[-2])):
        if event[0] >= frame_timestamp[i]:
            img_save_root = save_name + '/' + str(i).zfill(4) + '_' + str(sub_index) + '.png'
            if flag == False:
                flag = True
            else:
                hidden_state = hidden_pic_generator(pos_img, neg_img, last_pos_pic, last_neg_pic, hidden_state, stack_amount_3c, decay_rate_3c)
            last_pos_pic = pos_img
            last_neg_pic = neg_img
            save_2C_img(pos_img, neg_img, hidden_state, img_save_root)
            i += 1
            if i >= len(frame_timestamp):
                break
            sub_frame = np.linspace(frame_timestamp[i - 1], frame_timestamp[i], T_num)
            pos_img = np.full(pic_shape, 0, dtype=np.uint8)
            neg_img = np.full(pic_shape, 0, dtype=np.uint8)
            sub_index = 1

        elif event[0] < frame_timestamp[i]:
            if event[0] >= sub_frame[sub_index]:
                img_save_root = save_name + '/' + str(i).zfill(4) + '_' + str(sub_index) + '.png'
                if flag == False:
                    flag = True
                else:
                    hidden_state = hidden_pic_generator(pos_img, neg_img, last_pos_pic, last_neg_pic, hidden_state, stack_amount_3c, decay_rate_3c)
                last_pos_pic = pos_img
                last_neg_pic = neg_img
                save_2C_img(pos_img, neg_img, hidden_state, img_save_root)
                pos_img = np.full(pic_shape, 0, dtype=np.uint8)
                neg_img = np.full(pic_shape, 0, dtype=np.uint8)
                sub_index += 1
            process_event(pos_img, neg_img, hidden_state, event, pic_shape, stack_amount_1c2c)

    # Save last frame
    img_save_root = save_name + '/' + str(i).zfill(4) + '_' + str(sub_index) + '.png'
    hidden_state = hidden_pic_generator(pos_img, neg_img, last_pos_pic, last_neg_pic, hidden_state, stack_amount_3c, decay_rate_3c)
    save_2C_img(pos_img, neg_img, hidden_state, img_save_root)


def stack_dataset(root, stack_name, stack_amount_1c2c, stack_amount_3c, decay_rate_3c):
    """Process all sequences in a split folder."""
    text_root = os.path.join(root, "test.txt")
    file_name_list = []

    with open(text_root, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                file_name_list.append(line)

    for index, i in enumerate(sorted(file_name_list)):
        data = os.path.join(root, i).replace('\\', '/')
        if os.path.exists(join(data, stack_name).replace('\\', '/')):
            if 3 * len(os.listdir(join(data, 'img').replace('\\', '/'))) == len(
                    os.listdir(join(data, stack_name).replace('\\', '/'))):
                continue
        stack_event(stack_name, index, data, stack_amount_1c2c, stack_amount_3c, decay_rate_3c)


def parse_args():
    parser = argparse.ArgumentParser(description='GTP preprocessing for CSV event data')
    parser.add_argument('--source_dir', type=str, required=True, help='Root path of the dataset')
    parser.add_argument('--target_dir', type=str, required=True, help='Target dataset path (can be same as source)')
    parser.add_argument('--stack_name', type=str, default='inter1_stack_3008', help='Name of the output stack folder')
    parser.add_argument('--stack_amount_1c2c', type=float, default=30)
    parser.add_argument('--stack_amount_3c', type=float, default=30)
    parser.add_argument('--decay_rate_3c', type=float, default=0.8)
    return parser.parse_args()


args = parse_args()
target_dir = args.target_dir

train_root = os.path.join(target_dir, "train")
test_root = os.path.join(target_dir, "test")

stack_dataset(train_root, args.stack_name, args.stack_amount_1c2c, args.stack_amount_3c, args.decay_rate_3c)
stack_dataset(test_root, args.stack_name, args.stack_amount_1c2c, args.stack_amount_3c, args.decay_rate_3c)