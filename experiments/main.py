# %%
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

img = (
    ttf.to_tensor(Image.open("./samples/bulbul.jpeg").resize([W, H]))
    .unsqueeze(0)
    .cuda()
)


def param_fn(x):
    # flow = torch.zeros_like(x) + 1
    # flow[:, 1, 4] = torch.tensor([1, 3])
    # return flow
    return torch.tanh(x)


flow_layer = src.layers.Flow(H, W, parameterization=param_fn).to("cuda")
##src.utils.visualize_flow(flow_layer)
net = torchvision.models.resnet50(pretrained=True).cuda().eval()
optimizer = torch.optim.Adam(flow_layer.parameters(), lr=0.01)
tau = 0.005
target_class = 76
# %% w
adv_losses = []
flow_losses = []
for n in tqdm.tqdm(range(500)):
    flowed_img = flow_layer(img)
    out = net(normalize(flowed_img))[0]
    adv_loss = src.losses.adv_loss(out, target_class, -10)
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
# %%

plt.imshow(kornia.tensor_to_image(img))
plt.show()
plt.imshow(kornia.tensor_to_image(flow_layer(img)))
plt.show()
src.utils.visualize_flow(flow_layer, kornia.tensor_to_image(flowed_img))
# %%
