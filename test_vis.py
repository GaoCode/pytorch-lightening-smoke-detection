# import matplotlib.pyplot as plt
import src.util_fns as util_fns
import pickle
import numpy as np

# import cv2

# import os

# from pathlib import Path
# import xml.etree.ElementTree as ET
# import sklearn.metrics
# import torchvision
# import torch

from vis import display_image, display_bounding_box

metadata = pickle.load(open("./data/metadata.pkl", "rb"))
# print("metadata", metadata)

raw_data_path = "./data/"
raw_labels_path = "/userdata/kerasData/data/new_data/drive_clone/"
labels_path = "/root/pytorch_lightning_data/drive_clone_numpy/"

image = "data/20160604_FIRE_smer-tcs3-mobo-c/1465065908_+00240.jpg"

image_preds_path = (
    "./saved_logs/versions_246-258/version_246_hem-test/image_preds.csv"
)
output_path = "./data/final_split/hem-train_images_final.txt"

resize_dimensions = (1392, 1856)
crop_height = 1040
tile_dimensions = (224, 224)
tile_overlap = 20
num_tiles_height, num_tiles_width = util_fns.calculate_num_tiles(
    resize_dimensions, crop_height, tile_dimensions, tile_overlap
)


# create_hem_list(image_preds_path, output_path)

save_path = "data/img_smokey_net.jpg"
display_image(image, save_path=save_path)


save_path = "data/img_smokey_net_pred.jpg"
tile_probs_path = "data/tile_probs.npy"
tile_preds_path = "data/tile_preds.npy"
image_preds_path = "data/image_preds.npy"
probs = np.load(tile_probs_path).reshape((num_tiles_height, num_tiles_width))
image_pred = np.load(image_preds_path)
display_image(
    image, int(image_pred), "100", tile_probs=probs, idx=0, save_path=save_path
)

save_path = "data/img_smokey_net_gt_box.jpg"
image_name = image[5:-4]
gt_bboxes = metadata["bbox_labels"][image_name]
display_bounding_box(image, gt_bboxes, save_path=save_path)
