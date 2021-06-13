import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import parameter
import torchvision
from einops import rearrange, reduce, repeat
from PIL import Image

from flow import Flow


H, W = 224, 224

to_tensor = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize([H, W]),
    ]
)

normalize = torchvision.transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

img = Image.open("./bulbul.jpeg")

img = to_tensor(img).unsqueeze(0)


param_fn = lambda flow_before: torch.tanh(flow_before) * 1

flow_layer = Flow(
    height=H,
    width=W,
    parameterization=param_fn,
)

flowed_img = flow_layer(img)

from kornia import tensor_to_image as t2i

plt.imshow(t2i(flowed_img))
plt.show()
