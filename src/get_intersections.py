# -*- coding: utf-8 -*-

import sys
import glob
import cv2
import os
import numpy as np
import pandas as pd

from skimage.filters import threshold_sauvola
from skimage.morphology import disk, square, remove_small_objects, opening
from skimage.util import invert

# --- Image Processing pipeline ---

def RGB_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_threshold_sauvola(img):
    threshold_value = threshold_sauvola(img, 89, 0.15)
    return (img > threshold_value).astype(np.uint8)

def remove_small_obj(img, min_size):
    return remove_small_objects(img < 1, min_size=min_size)

def apply_opening(img, footprint):
    return opening(img, footprint)

def compute_connected_components(img):
    return cv2.connectedComponents(img.astype(np.uint8))[1]

def apply_preprocessing(img):
    img_list = [img]

    img_list.append(RGB_to_gray(img_list[-1]))

    img_list.append(apply_threshold_sauvola(img_list[-1]))

    img_list.append(apply_opening(invert(img_list[-1]), square(1)))

    img_list.append(remove_small_obj(invert(img_list[-1]), 20000))

    img_list.append(remove_small_obj(img_list[-1], 1000))

    img_list.append(apply_opening(img_list[-1], disk(10)))

    img_list.append(compute_connected_components(img_list[-1]))

    return img_list[-1]

def detect_intersection(img, window_size, min_count=20):
    shape = img.shape
    h_window_size = window_size // 2
    intersections = pd.DataFrame(columns=[0, 1])

    for i in range(0, shape[0] - window_size, h_window_size):
        for j in range(0, shape[1] - window_size, h_window_size):
            img_window = img[i:i+window_size, j:j+window_size]

            _, count = np.unique(img_window, return_counts=True)
            # Check if img_window contains at least 3 colors and each component contains more than min_count points
            if len(count[count > min_count]) > 3:
                intersections.loc[intersections.shape[0]] = [i + h_window_size, j + h_window_size]

    return intersections

def build_intersections_image(img, intersections):
    img_zeros = np.zeros_like(img)
    for [x, y] in intersections.values.tolist():
        img_zeros[x, y] = 1
    return img_zeros

def merge_intersection_points(img, intersections, window_size, start_idx=0):
    shape = img.shape
    h_window_size = window_size // 2
    new_intersections = pd.DataFrame(columns=[0, 1])

    img_zeros = build_intersections_image(img, intersections)

    for i in range(start_idx, shape[0] - window_size, window_size):
        for j in range(start_idx, shape[1] - window_size, window_size):
            img_window = img_zeros[i:i+window_size, j:j+window_size]

            components, _ = np.unique(img_window, return_counts=True)
            # Check if img_window contains 0 and 1
            if len(components) == 2:
                indices = np.argwhere(img_window == 1)
                mean_x = int(np.mean(indices[:, 0]))
                mean_y = int(np.mean(indices[:, 1]))

                new_intersections.loc[new_intersections.shape[0]] = [i + mean_x, j + mean_y]

    return new_intersections

# --------------------------------

def load_file(path):
    return cv2.imread(path)

def save_results(df, path):
    df.to_csv(path, sep=',', header=False, index=False)

def process_images(images_path):
    if not os.path.exists("RESULTS/"):
        os.mkdir('RESULTS/')

    for img_path in images_path:
        print("Load: {}".format(img_path))

        # load the image
        img = load_file(img_path)

        # process the image
        preprocessed_img = apply_preprocessing(img)

        # get intersection points
        intersections = detect_intersection(preprocessed_img, 70)
        new_intersections = merge_intersection_points(preprocessed_img, intersections, 100)
        new_intersections = merge_intersection_points(preprocessed_img, new_intersections, 80, start_idx=40)

        # save the results
        img_name = img_path.split('/')[-1][:-4]
        csv_path = 'RESULTS/' + img_name + '.csv'
        print("Save: {}".format(csv_path))
        print()
        save_results(new_intersections, csv_path)

def main(argv):
    if len(argv) != 2:
        print('Usage: python3 get_intersections.py <path_to_images>')
        return

    path = argv[1]
    path = path if path[-1] == '/' else path + '/'
    images_path = glob.glob(path + '*.jpg')
    process_images(images_path)

if __name__ == "__main__":
    main(sys.argv)
