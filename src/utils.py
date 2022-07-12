# %%
import torch, torchvision
import matplotlib.pyplot as plt
import PIL
from torch.utils.data import Dataset
import os
import csv
import kornia
from piqa.tv import tv
import numpy as np


# def visualize_flow(flow_layer, image=None, grid=False, figsize=[15, 15]):
#     H = flow_layer.H
#     W = flow_layer.W
#     with torch.no_grad():
#         flow = flow_layer.get_applied_flow().cpu().numpy()
#     if image is None:
#         image = np.ones(shape=[flow_layer.H, flow_layer.W, 3])
#     plt.figure(figsize=figsize)

#     if grid:
#         ax = plt.gca()
#         ax.set_xticks(np.arange(-0.5, W, 1))
#         ax.set_yticks(np.arange(-0.5, H, 1))
#         ax.set_xticklabels(np.arange(1, H + 2, 1))
#         ax.set_yticklabels(np.arange(1, W + 2, 1))
#         plt.grid()
#         plt.imshow(image)

#     plt.quiver(
#         flow[1], flow[0], units="xy", angles="xy", scale=1,
#     )
#     plt.show()


def t_imshow(img: torch.Tensor, save_str: str = ""):
    img_np = kornia.utils.tensor_to_image(img)
    plt.imshow(img_np)
    plt.show()
    if save_str:
        plt.imsave(save_str, img_np)


def load_transfer_net():
    normalize = torchvision.transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )
    net = torchvision.models.resnet50(pretrained=True)
    # net.load_state_dict(torch.load("models/inception_v3_google-1a9a5a14.pth"))
    net = torch.nn.Sequential(normalize, net)
    net = net.cuda().eval()
    for param in net.parameters():
        param.requires_grad = False
    return net


def load_net():
    normalize = torchvision.transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )
    net = torchvision.models.inception_v3(pretrained=True)
    # net.load_state_dict(torch.load("models/inception_v3_google-1a9a5a14.pth"))
    net = torch.nn.Sequential(normalize, net)
    net = net.cuda().eval()
    for param in net.parameters():
        param.requires_grad = False
    return net


def flow_cbcr(image, flow_layer):
    image = torch.repeat_interleave(image, flow_layer.batch_size, 0)
    img_yuv = kornia.color.rgb_to_yuv(image)
    img_y = img_yuv[:, :1, :, :]
    img_uv = img_yuv[:, 1:, :, :]
    flowed_img = flow_layer(img_uv)
    flowed_img = torch.cat([img_y, flowed_img], dim=-3)
    flowed_rgb = kornia.color.yuv_to_rgb(flowed_img)
    return torch.clamp(flowed_rgb, 0, 1)


def flow_y(image, flow_layer):
    image = torch.repeat_interleave(image, flow_layer.batch_size, 0)
    img_lab = kornia.color.rgb_to_lab(image)

    img_l = img_lab[:, :1, :, :]
    img_ab = img_lab[:, 1:, :, :]
    flowed_img = flow_layer(img_l)
    flowed_img = torch.cat([flowed_img, img_ab], dim=-3)
    flowed_rgb = kornia.color.lab_to_rgb(flowed_img)
    return torch.clamp(flowed_rgb, 0, 1)


# def flow_h(image, flow_layer):
#     image = torch.repeat_interleave(image, flow_layer.batch_size, 0)
#     img_hsv = kornia.color.rgb_to_hsv(image)
#     img_h = img_hsv[:, :1, :, :]
#     img_sv = img_hsv[:, 1:, :, :]
#     flowed_img = flow_layer(img_h)
#     flowed_img = torch.cat([flowed_img, img_sv], dim=-3)
#     return kornia.color.hsv_to_rgb(flowed_img)


def flow_l(image, flow_layer):
    image = torch.repeat_interleave(image, flow_layer.batch_size, 0)
    img_lab = kornia.color.rgb_to_lab(image)

    img_l = img_lab[:, :1, :, :]
    img_ab = img_lab[:, 1:, :, :]
    flowed_img = flow_layer(img_l)
    flowed_img = torch.cat([flowed_img, img_ab], dim=-3)
    flowed_rgb = kornia.color.lab_to_rgb(flowed_img)
    return torch.clamp(flowed_rgb, 0, 1)


def flow_ab(image, flow_layer):
    image = torch.repeat_interleave(image, flow_layer.batch_size, 0)
    img_lab = kornia.color.rgb_to_lab(image)

    img_l = img_lab[:, :1, :, :]
    img_ab = img_lab[:, 1:, :, :]
    flowed_img = flow_layer(img_ab)
    flowed_img = torch.cat([img_l, flowed_img], dim=-3)
    flowed_rgb = kornia.color.lab_to_rgb(flowed_img)
    return torch.clamp(flowed_rgb, 0, 1)


def flow_rgb(image, flow_layer):
    image = torch.repeat_interleave(image, flow_layer.batch_size, 0)
    flowed_img = flow_layer(image)
    return torch.clamp(flowed_img, 0, 1)


class NIPS2017TargetedDataset(Dataset):
    def __init__(self, dataset_path, head=None):
        self.head = head
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

        if head:
            self.images = self.images[:head]
            self.true_classes = self.true_classes[:head]
            self.target_classes = self.target_classes[:head]

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


def flow_loss(flow_layer):
    return tv(flow_layer.get_applied_flow())


def adv_loss(adv_logits_batch, target_class, K):
    """
    adv_logits_batch: [N, C]
    target_class: int
    kappa: non-negative float

    Returns: a list of loss

    """
    assert K >= 0

    losses = torch.zeros_like(adv_logits_batch[:, 0])
    for n in range(len(adv_logits_batch)):
        adv_logits = adv_logits_batch[n]
        top_two_logits, top_two_classes = torch.topk(adv_logits, 2)
        target_class_logit = adv_logits[target_class]

        if top_two_classes[0] == target_class:
            nontarget_max = top_two_logits[1]
        else:
            nontarget_max = top_two_logits[0]

        loss = torch.maximum(nontarget_max - target_class_logit, torch.tensor(-K))
        losses[n] = loss
    return losses, losses.sum()


def image_colorfulness(image):
    # split the image into its respective RGB components
    (R, G, B) = image[..., 0], image[..., 1], image[..., 2]
    # compute rg = R - G
    rg = np.absolute(R - G)
    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)
    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    # derive the "colorfulness" metric and return it
    colorfulness = stdRoot + (0.3 * meanRoot)

    return stdRoot, meanRoot, colorfulness
