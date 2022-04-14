# %%
from src.losses import flow_loss
import src
import torch, torchvision
import matplotlib.pyplot as plt
import PIL
import numpy as np
import tqdm
from torch.utils.data import Dataset
import os
import csv
import kornia


def visualize_flow(flow_layer, image=None, grid=False, figsize=[15, 15]):
    H = flow_layer.H
    W = flow_layer.W
    with torch.no_grad():
        flow = flow_layer.get_applied_flow().cpu().numpy()
    if image is None:
        image = np.ones(shape=[flow_layer.H, flow_layer.W, 3])
    plt.figure(figsize=figsize)

    if grid:
        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, W, 1))
        ax.set_yticks(np.arange(-0.5, H, 1))
        ax.set_xticklabels(np.arange(1, H + 2, 1))
        ax.set_yticklabels(np.arange(1, W + 2, 1))
        plt.grid()
        plt.imshow(image)

    plt.quiver(
        flow[1], flow[0], units="xy", angles="xy", scale=1,
    )
    plt.show()


def flow_uv(image, flow_layer):
    img_yuv = kornia.color.rgb_to_yuv(image)
    img_y = img_yuv[:, :1, :, :]
    img_uv = img_yuv[:, 1:, :, :]
    uv_flowed_img = flow_layer(img_uv)
    flowed_img = torch.cat([img_y, uv_flowed_img], dim=-3)

    return kornia.color.yuv_to_rgb(flowed_img)


def flow_h(image, flow_layer):
    img_hsv = kornia.color.rgb_to_hsv(image)
    img_h = img_hsv[:, :1, :, :]
    img_sv = img_hsv[:, 1:, :, :]
    h_flowed_img = flow_layer(h)
    flowed_img = torch.cat([h_flowed_img, img_sv], dim=-3)

def flow_l(image, flow_layer):


class NIPS2017TargetedDataset(Dataset):
    def __init__(self, dataset_path):
        labels_csv_path = os.path.join(dataset_path, "images.csv")
        with open(labels_csv_path) as csvfile:
            labels_list = list(csv.reader(csvfile))[1:]

        self.image_names = [f"{row[0]}.png" for row in labels_list]
        image_paths = [
            os.path.join(dataset_path, "images", name) for name in self.image_names
        ]
        self.images = [PIL.Image.open(path) for path in image_paths]
        self.true_classes = [int(row[6]) for row in labels_list]
        self.target_classes = [int(row[7]) for row in labels_list]

        categories_csv_path = os.path.join(dataset_path, "categories.csv")
        with open(categories_csv_path) as csvfile:
            categories_list = list(csv.reader(csvfile))[1:]
        self.class_names = [row[1] for row in categories_list]

        assert len(self.images) == len(self.true_classes) == len(self.target_classes)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, n):
        image_name = self.image_names[n][:-4]  # discard ".png"
        image_tensor = torchvision.transforms.functional.to_tensor(self.images[n])

        # since dataset has dummy class 0 and all labels are shifted
        true_class = self.true_classes[n] - 1
        target_class = self.target_classes[n] - 1

        true_class_name = self.class_names[true_class]
        target_class_name = self.class_names[target_class]

        return {
            "image": image_tensor,
            "image_name": image_name,
            "true_class": true_class,
            "target_class": target_class,
            "true_class_name": true_class_name,
            "target_class_name": target_class_name,
        }


