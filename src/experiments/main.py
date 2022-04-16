#  %%
from src.utils import visualize_flow
import torch, torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import kornia
import os
import tqdm
import torchvision.transforms.functional as ttf
import src

H, W = 300, 300

normalize = torchvision.transforms.Normalize(
    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
)

img = ttf.to_tensor(Image.open("./samples/2400.png").resize([W, H])).unsqueeze(0).cuda()


def param_fn(x):
    return torch.tanh(5 * x) * 10


rgb_flow_layer = src.layers.Flow(H, W, param=param_fn).to("cuda")
hsv_flow_layer = src.layers.Flow(H, W, param=param_fn).to("cuda")
yuv_flow_layer = src.layers.Flow(H, W, param=param_fn).to("cuda")
lab_flow_layer = src.layers.Flow(H, W, param=param_fn).to("cuda")

rgb_optimizer = torch.optim.Adam(rgb_flow_layer.parameters(), lr=0.01)
hsv_optimizer = torch.optim.Adam(hsv_flow_layer.parameters(), lr=0.01)
yuv_optimizer = torch.optim.Adam(yuv_flow_layer.parameters(), lr=0.01)
lab_optimizer = torch.optim.Adam(lab_flow_layer.parameters(), lr=0.01)


net = torchvision.models.resnet50(pretrained=True).cuda().eval()
tau = 0
target_class = 76
adv_losses = []
flow_losses = []
for n in tqdm.tqdm(range(100)):
    flowed_img = rgb_flow_layer(img)
    h_flowed_img = src.utils.flow_h(img, hsv_flow_layer)
    uv_flowed_img = src.utils.flow_uv(img, yuv_flow_layer)
    ab_flowed_img = src.utils.flow_ab(img, lab_flow_layer)

    imglist = [h_flowed_img, uv_flowed_img, ab_flowed_img]
    optlist = [hsv_optimizer, yuv_optimizer, lab_optimizer]
    layerlist = [rgb_flow_layer, hsv_flow_layer, yuv_flow_layer, lab_flow_layer]

    for flowed_img, optimizer, flow_layer in zip(imglist, optlist, layerlist):
        out = net(normalize(flowed_img))[0]
        adv_loss = src.losses.adv_loss(out, target_class, 0)
        flow_loss = src.losses.flow_loss(flow_layer)
        adv_losses.append(adv_loss.item())
        flow_losses.append(flow_loss.item())

        loss = adv_loss + tau * flow_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

plt.plot(adv_losses)
plt.show()
plt.plot(flow_losses)
plt.show()


def max_contrast(img):
    img = img - img.min()
    img = img / img.max()
    return img


# %%

plt.imshow(kornia.tensor_to_image(img))
plt.show()
for flowed_img in imglist:
    plt.imshow(kornia.tensor_to_image(abs(flowed_img)))
    plt.show()


# %%
