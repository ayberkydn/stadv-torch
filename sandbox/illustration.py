# %%
import torch, torchvision
import piqa
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import kornia
import os
import tqdm
import torchvision.transforms.functional as ttf
import src
from src.layers import Flow
from src.utils import flow_ab, flow_cbcr, flow_rgb, t_imshow
import importlib


def luma_chroma(img: np.ndarray):
    img = kornia.image_to_tensor(img)
    img_ycbcr = kornia.color.rgb_to_ycbcr(img)
    img_y, img_cbcr = img_ycbcr[0:1, :, :], img_ycbcr[1:, :, :]
    img_y, img_cbcr = (
        torch.cat([img_y, torch.ones_like(img_cbcr) / 2]),
        torch.cat([torch.ones_like(img_y) / 2, img_cbcr]),
    )
    img_y, img_cbcr = (
        kornia.color.ycbcr_to_rgb(img_y),
        kornia.color.ycbcr_to_rgb(img_cbcr),
    )
    img_y, img_cbcr = torch.clamp(img_y, 0, 1), torch.clamp(img_cbcr, 0, 1)
    img_y, img_cbcr = kornia.tensor_to_image(img_y), kornia.tensor_to_image(img_cbcr)
    return img_y, img_cbcr


name = "f6437ed4f1b8fe23"
for data in src.utils.NIPS2017TargetedDataset("./data"):
    if data["image_name"] != name:
        continue
    else:
        flow = Flow(299, 299, init_std=5)
        img_tensor = data["image"].unsqueeze(0)
        img = kornia.tensor_to_image(img_tensor)
        img_y, img_cbcr = luma_chroma(img)
        img_flowed = kornia.tensor_to_image(flow_ab(img_tensor, flow))
        img_flowed_rgb = kornia.tensor_to_image(flow_rgb(img_tensor, flow))
        img_flowed_y, img_flowed_cbcr = luma_chroma(img_flowed_rgb)

        startx, starty = 100, 100
        startx, starty = starty, startx
        delta = 100

        # img: original image
        # img_y : y of image
        # img_flowed: colorspace flowed image
        # img_flowed_cbcr: cbcr of img_flowed

        # plt.axis("off")
        # plt.imshow(img)
        # plt.savefig("figures/illus/benign.png", bbox_inches="tight")
        # plt.show()

        # plt.axis("off")
        # plt.imshow(img[startx : startx + delta, starty : starty + delta, :])
        # plt.savefig("figures/illus/benign_crop.png", bbox_inches="tight")
        # plt.show()

        # plt.axis("off")
        # plt.imshow(img_y)
        # plt.savefig("figures/illus/benign_y.png", bbox_inches="tight")
        # plt.show()

        # plt.axis("off")
        # plt.imshow(img_cbcr)
        # plt.savefig("figures/illus/benign_cbcr.png", bbox_inches="tight")
        # plt.show()

        # plt.axis("off")
        # plt.imshow(img_cbcr[startx : startx + delta, starty : starty + delta, :])
        # plt.savefig("figures/illus/benign_cbcr_crop.png", bbox_inches="tight")
        # plt.show()

        # plt.axis("off")
        # plt.imshow(img_flowed_cbcr)
        # plt.savefig("figures/illus/adversarial_cbcr.png", bbox_inches="tight")
        # plt.show()

        # plt.axis("off")
        # plt.imshow(img_flowed_cbcr[startx : startx + delta, starty : starty + delta, :])
        # plt.savefig("figures/illus/adversarial_cbcr_crop.png", bbox_inches="tight")
        # plt.show()

        plt.axis("off")
        plt.imshow(img_flowed)
        plt.savefig("figures/illus/adversarial.png", bbox_inches="tight")
        plt.show()

        # plt.axis("off")
        # plt.imshow(img_flowed[startx : startx + delta, starty : starty + delta, :])
        # plt.savefig("figures/illus/adversarial_crop.png", bbox_inches="tight")
        # plt.show()

        break
# plt.imshow(flowed_img_np[start:end, start:end, :])
# plt.show()

