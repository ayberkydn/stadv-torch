# %%
import torch, torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import kornia
import stadv_torch

H, W = 224, 224


to_tensor = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Resize([H, W])]
)

normalize = torchvision.transforms.Normalize(
    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
)


img = to_tensor(Image.open("./samples/bulbul.jpeg")).unsqueeze(0).cuda()


flow_layer = stadv_torch.layers.Flow(H, W).to("cuda")
optimizer = torch.optim.Adam(flow_layer.parameters(), lr=0.1)

# %%
net = torchvision.models.resnet50(pretrained=True).cuda().eval()

# %%
import tqdm


for n in tqdm.tqdm(range(1000)):
    out = net(normalize(img))
    adv_loss = stadv_torch.losses.adv_loss(out, 5, 0)
