#%%
import src
import torch, torchvision
from src.utils import NIPS2017TargetedDataset
import tqdm
import matplotlib.pyplot as plt

#%%

dset = NIPS2017TargetedDataset("data/nips2017_targeted")

net = torchvision.models.inception_v3().cuda()
net.load_state_dict(torch.load("models/inception_v3_google-1a9a5a14.pth"))
net.eval()


data = dset[0]
#%%

img = data["image"].cuda().unsqueeze(0)
true_class = data["true_class"]
target_class = data["target_class"]

flow_layer = src.layers.Flow(299, 299, parameterization=None).to("cuda")
optimizer = torch.optim.Adam(flow_layer.parameters(), lr=0.1)
#%%

K = 0
for n in tqdm.tqdm(range(200)):
    # flowed_img = src.utils.flow_uv(img, flow_layer)
    flowed_img = flow_layer(img)
    out = net(flowed_img * 2 - 1)[0]
    adv_loss = src.losses.adv_loss(out, target_class, K)
    loss = adv_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
