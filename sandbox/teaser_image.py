#%%
import src
import torch, torchvision
from src.utils import NIPS2017TargetedDataset
import tqdm
import matplotlib.pyplot as plt
from kornia import tensor_to_image as t2i

#%%

dset = NIPS2017TargetedDataset("data/nips2017_targeted")

net = torchvision.models.inception_v3().cuda()
net.load_state_dict(torch.load("models/inception_v3_google-1a9a5a14.pth"))
net.eval()


data = dset[0]
#%%

for n in range(1000):
    data = dset[n]
    if data["image_name"] == "73a52afd2f818ed5":
        img = data["image"].cuda().unsqueeze(0)
        true_class = data["true_class"]
        target_class = data["target_class"]
        print(n)
        break

plt.imshow(t2i(img))
plt.show()


#%% stadv

flow_layer = src.layers.Flow(299, 299, param=None).to("cuda")
optimizer = torch.optim.Adam(flow_layer.parameters(), lr=0.1)
K = -2
tau = 0.0025
for n in tqdm.tqdm(range(200)):
    # flowed_img = src.utils.flow_uv(img, flow_layer)
    stadv_flowed_img = flow_layer(img)
    out = net(stadv_flowed_img * 2 - 1)[0]
    adv_loss = src.losses.adv_loss(out, target_class, K)
    flow_loss = src.losses.flow_loss(flow_layer)
    loss = adv_loss + tau * flow_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(adv_loss)
plt.imshow(t2i(stadv_flowed_img))
plt.show()


# %% bizim atak
flow_layer = src.layers.Flow(299, 299, param=None).to("cuda")
optimizer = torch.optim.Adam(flow_layer.parameters(), lr=0.1)
K = 0
for n in tqdm.tqdm(range(200)):
    # flowed_img = src.utils.flow_uv(img, flow_layer)
    uv_flowed_img = src.utils.flow_uv(img, flow_layer)
    out = net(uv_flowed_img * 2 - 1)[0]
    adv_loss = src.losses.adv_loss(out, target_class, K)
    loss = adv_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.imshow(t2i(uv_flowed_img))
plt.show()

#%%

zoom_x, zoom_y = 100, 150
delta = 50

uv_flowed_img = torch.clamp(uv_flowed_img, 0, 1)

original_zoomed = img[0, :, zoom_y : zoom_y + delta, zoom_x : zoom_x + delta]
uv_flowed_zoomed = uv_flowed_img[0, :, zoom_y : zoom_y + delta, zoom_x : zoom_x + delta]
stadv_flowed_zoomed = stadv_flowed_img[
    0, :, zoom_y : zoom_y + delta, zoom_x : zoom_x + delta
]

from kornia import resize

plt.imsave("saved_images/original.png", t2i(img))
# plt.imsave("saved_images/stadv_flowed.png", t2i(stadv_flowed_img))
# plt.imsave("saved_images/uv_flowed.png", t2i(uv_flowed_img))
plt.imsave(
    "saved_images/zoom_original.png", t2i(resize(original_zoomed, 500, "nearest"))
)
plt.imsave(
    "saved_images/zoom_stadv_flowed.png",
    t2i(resize(stadv_flowed_zoomed, 500, "nearest")),
)
plt.imsave(
    "saved_images/zoom_uv_flowed.png", t2i(resize(uv_flowed_zoomed, 500, "nearest"))
)

# %%
