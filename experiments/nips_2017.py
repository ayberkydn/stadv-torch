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

dataset = src.utils.NIPS2017TargetedDataset("data/nips2017_targeted")

net = torchvision.models.inception_v3().cuda()
net.load_state_dict(torch.load("models/inception_v3_google-1a9a5a14.pth"))
net.eval()

flow_layer = src.layers.Flow(299, 299).to("cuda")

optimizer = torch.optim.Adam(flow_layer.parameters(), lr=0.01)
tau = 0.002
# %%
adv_losses = []
flow_losses = []
data = dataset[0]
true_class = data["true_class"]
target_class = data["target_class"]
img = data["image"].cuda().unsqueeze(0)
for n in tqdm.tqdm(range(300)):
    flowed_img = flow_layer(img)
    out = net(flowed_img * 2 - 1)[0]
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

plt.figure(figsize=[15, 15])
plt.imshow(kornia.tensor_to_image(img))
plt.show()
plt.figure(figsize=[15, 15])
plt.imshow(kornia.tensor_to_image(flow_layer(img)))
plt.show()
#%%
# src.utils.visualize_flow(flow_layer, kornia.tensor_to_image(flowed_img))

# %%
