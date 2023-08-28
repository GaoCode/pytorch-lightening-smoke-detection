import matplotlib.pyplot as plt
import src.util_fns as util_fns

# import pickle
import numpy as np
import cv2
import os
from pathlib import Path

# import xml.etree.ElementTree as ET
# import sklearn.metrics
import torchvision
import torch


def calculate_overlap_ticks(max_dim, tile_size=224, tile_overlap=20):
    i = 0
    dim = 0
    ticks = []

    while dim < max_dim:
        if i == 0:
            dim += tile_size - tile_overlap
        elif i % 2 == 1:
            dim += tile_overlap
        elif i % 2 == 0:
            dim += tile_size - tile_overlap * 2

        ticks.append(dim)
        i += 1

    return ticks


def create_hem_list(image_preds_path, output_path):
    """Description: Creates txt file of hardest positive and negative examples sorted by image_loss"""

    csv = np.loadtxt(image_preds_path, delimiter=",", dtype=str)

    positive_images = {}
    negative_images = {}

    for image, pred, loss in csv:
        if util_fns.get_ground_truth_label(image) == 1:
            positive_images[image] = float(loss)
        else:
            negative_images[image] = float(loss)

    num_to_keep = (len(positive_images) + len(negative_images)) // 10

    hem_list = (
        sorted(negative_images, key=negative_images.get)[-num_to_keep:]
        + sorted(positive_images, key=positive_images.get)[-num_to_keep:]
    )

    np.savetxt(output_path, hem_list, fmt="%s")


def display_image(
    image_path,
    label_image_path="",
    image_pred=None,
    image_prob=None,
    tile_preds=None,
    tile_probs=None,
    idx=0,
    save_path=None,
    resize_dimensions=(1392, 1856),
    crop_height=1040,
    tile_dimensions=(224, 224),
    tile_overlap=20,
    grid=True,
):
    """
    View multiple images stored in files, stacking vertically

    Arguments:
        filename: str - path to filename containing image
    """
    # image_path = raw_data_path + image_name + ".jpg"
    image_name = os.path.basename(image_path)[:-4]

    # label_image_path = (
    #     raw_labels_path
    #     + util_fns.get_fire_name(image_name)
    #     + "/labels/"
    #     + util_fns.get_only_image_name(image_name)
    #     + ".jpg"
    # )

    gt_label = util_fns.get_ground_truth_label(image_name)
    print("image_path", image_path)

    # if Path(label_image_path).exists():
    #     filename = label_image_path
    # el
    if Path(image_path).exists():
        filename = image_path
    else:
        print("No image to display.")
        return

    # Load and process image
    print("filename", filename)
    img = cv2.imread(filename)
    img = cv2.resize(img, (resize_dimensions[1], resize_dimensions[0]))[
        -crop_height:
    ]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display image
    plt.figure(figsize=(16, 12), dpi=80)

    plt.title("ID: " + str(idx), loc="left")
    plt.title(
        image_name, loc="center", color="black" if gt_label == 0 else "red"
    )
    if image_pred is not None:
        is_correct = image_pred == gt_label
        plt.title(
            "IMAGE_PROB={}, IMAGE_PRED={}".format(image_prob, image_pred) + " ",
            loc="right",
            color="black" if is_correct else "red",
        )

    plt.imshow(img)

    # Calculate grid ticks
    if grid:
        plt.xticks(
            calculate_overlap_ticks(
                resize_dimensions[1], tile_dimensions[1], tile_overlap
            ),
            alpha=0,
        )
        plt.yticks(
            calculate_overlap_ticks(
                crop_height, tile_dimensions[0], tile_overlap
            ),
            alpha=0,
        )
        plt.grid()
    else:
        plt.xticks([])
        plt.yticks([])

    # Insert tile preds text
    if tile_preds is not None or tile_probs is not None:
        x_text_ticks = np.insert(
            np.arange(
                tile_dimensions[1],
                resize_dimensions[1],
                tile_dimensions[1] - tile_overlap,
            ),
            0,
            0,
        )
        y_text_ticks = (
            np.insert(
                np.arange(
                    tile_dimensions[0],
                    crop_height,
                    tile_dimensions[0] - tile_overlap,
                ),
                0,
                0,
            )
            + tile_overlap
        )

        for i, x in enumerate(x_text_ticks):
            for j, y in enumerate(y_text_ticks):
                if tile_preds is not None:
                    plt.text(
                        x,
                        y,
                        tile_preds[j, i],
                        size="medium",
                        weight="heavy",
                        color="white" if tile_preds[j, i] == 0 else "red",
                    )
                else:
                    plt.text(
                        x,
                        y,
                        round(tile_probs[j, i], 2),
                        size="medium",
                        weight="heavy",
                        color="white" if tile_probs[j, i] < 0.5 else "red",
                    )

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


def display_bounding_box(
    image,
    gt_bboxes,
    resize_dimensions=(1392, 1856),
    crop_height=1040,
    save_path=None,
):
    label_image_path = image
    img = cv2.imread(label_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resize_factor = np.array(resize_dimensions) / np.array(img.shape[:2])

    img = cv2.resize(img, (resize_dimensions[1], resize_dimensions[0]))[
        -crop_height:
    ]
    img = img.transpose(2, 0, 1)

    bboxes = []
    # gt_bboxes = metadata["bbox_labels"][image_name]
    for i in range(len(gt_bboxes)):
        bboxes.append([0, 0, 0, 0])
        bboxes[i][0] = gt_bboxes[i][0] * resize_factor[1]
        bboxes[i][1] = gt_bboxes[i][1] * resize_factor[0] - (
            resize_dimensions[0] - crop_height
        )
        bboxes[i][2] = gt_bboxes[i][2] * resize_factor[1]
        bboxes[i][3] = gt_bboxes[i][3] * resize_factor[0] - (
            resize_dimensions[0] - crop_height
        )

        bboxes[i][1] = np.maximum(bboxes[i][1], 0)
        bboxes[i][3] = np.maximum(bboxes[i][3], 1)

    imgs = torchvision.utils.draw_bounding_boxes(
        torch.as_tensor(img, dtype=torch.uint8), torch.as_tensor(bboxes)
    )

    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(16, 12))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
