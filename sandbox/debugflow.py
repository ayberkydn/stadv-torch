#%%
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
from src.layers import Flow, SpatialAttacker
from src.utils import flow_rgb, t_imshow
import importlib

data = src.utils.NIPS2017TargetedDataset("./data")[4]
img = data["image"].unsqueeze(0).cuda()
net = src.utils.load_net()
param = lambda x: torch.tanh(x + 5) + 150
# param = torch.nn.Tanh()
flow = src.layers.Flow(299, 299, 1, 1, param).cuda()
flowed_img = flow_rgb(img, flow)
t_imshow(flowed_img)

# %%
attacker = SpatialAttacker(img, net, 16, "rgb", is_restricted=False)
attacker.attack_targeted(64, 1000, 2000, 0.01)
t_imshow(attacker.get_adversarial_tensor())
#%%
flowed_img = attacker.flow(img)
t_imshow(flowed_img)

