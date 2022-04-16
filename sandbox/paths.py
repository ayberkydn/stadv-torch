#%%
import os
import sys

import torch, torchvision
from torchvision import transforms
import src
from torch.utils.data import DataLoader
import tqdm
import matplotlib.pyplot as plt
import kornia
import random

dataset = src.utils.NIPS2017TargetedDataset("data/nips2017_targeted")

net = torchvision.models.inception_v3().cuda()
net.load_state_dict(torch.load("models/inception_v3_google-1a9a5a14.pth"))
net.eval()

data = dataset[10]

param_fn = lambda x: torch.tanh(x)
# param_fn = lambda x: x
flow_layer = src.layers.Flow(299, 299, param=param_fn).to("cuda")

optimizer = torch.optim.Adam(flow_layer.parameters(), lr=0.05)

true_class = data["true_class"]
target_class = data["target_class"]
img = data["image"].cuda().unsqueeze(0)
tau = 0.0
for n in tqdm.tqdm(range(200)):
    optimizer.zero_grad()
    # flowed_img = flow_layer(img)
    flowed_img = src.utils.flow_uv(img, flow_layer)
    out = net(flowed_img * 2 - 1)[0]
    adv_loss = src.losses.adv_loss(out, target_class, 0)
    flow_loss = src.losses.flow_loss(flow_layer)
    loss = adv_loss + tau * flow_loss
    loss.backward()
    if loss.item() == 0:
        break
    optimizer.step()


#%%

flowed_img = src.utils.flow_uv(img, flow_layer)
plt.figure(figsize=[15, 15])
plt.imshow(kornia.tensor_to_image(img))
plt.show()
plt.figure(figsize=[15, 15])
plt.imshow(kornia.tensor_to_image(flowed_img))
plt.show()
# %%
