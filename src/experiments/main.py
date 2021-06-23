# %%
import torch, torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import kornia
import os
import tqdm
import torchvision.transforms.functional as ttf

import stadv_torch

H, W = 150, 150

normalize = torchvision.transforms.Normalize(
    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
)

img = (
    ttf.to_tensor(Image.open("./samples/bulbul.jpeg").resize([H, W]))
    .unsqueeze(0)
    .cuda()
)

flow_layer = stadv_torch.layers.Flow(H, W).to("cuda")
net = torchvision.models.resnet50(pretrained=True).cuda().eval()
optimizer = torch.optim.Adam(flow_layer.parameters(), lr=0.01)
tau = 0.05
# %%
adv_losses = []
flow_losses = []
for n in tqdm.tqdm(range(200)):
    flowed_img = flow_layer(img)
    out = net(normalize(flowed_img))[0]
    target_class = 7
    adv_loss = stadv_torch.losses.adv_loss(out, target_class, -10)
    flow_loss = stadv_torch.losses.flow_loss(flow_layer)
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

# %%
