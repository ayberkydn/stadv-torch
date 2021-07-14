from src.layers import Flow
import src
import kornia
from kornia.utils import tensor_to_image as t2i
import matplotlib.pyplot as plt
import torch

dataset = src.utils.NIPS2017TargetedDataset("data/nips2017_targeted")
for data in dataset:
    if data["image_name"] == "01244097ca8ffdfa":
        break

image = data["image"].unsqueeze(0)
image_yuv = kornia.color.rgb_to_yuv(image)


def param_fn(x):
    return torch.tanh(x)
    # return x * 5


flow_layer = Flow(299, 299, parameterization=param_fn)

image_y = image_yuv[:, :1, :, :]
image_uv = image_yuv[:, 1:, :, :]

showed_uv = torch.cat([torch.ones_like(image_y) * 0.5, image_uv], dim=-3)
showed_uv = torch.clamp(kornia.color.yuv_to_rgb(showed_uv), 0, 1)


distorted_uv = flow_layer(showed_uv)

distorted_image_yuv = torch.cat([image_y, image_uv], dim=-3)
distorted_image_rgb = torch.clamp(kornia.color.yuv_to_rgb(distorted_image_yuv), 0, 1)


#%% zoom
zoom_y = 75
zoom_x = 220
delta = 50

z_img = image[:, :, zoom_y : zoom_y + delta, zoom_x : zoom_x + delta]
z_uv = showed_uv[:, :, zoom_y : zoom_y + delta, zoom_x : zoom_x + delta]
z_distorted_uv = distorted_uv[:, :, zoom_y : zoom_y + delta, zoom_x : zoom_x + delta]
z_distorted_image = distorted_image_rgb[
    :, :, zoom_y : zoom_y + delta, zoom_x : zoom_x + delta
]

z_img = kornia.resize(z_img, [299, 299])
z_uv = kornia.resize(z_uv, [299, 299])
z_distorted_uv = kornia.resize(z_distorted_uv, [299, 299])
z_distorted_image = kornia.resize(z_distorted_image, [299, 299])
#%%
plt.imshow(t2i(image))
plt.imshow(t2i(z_uv))
#%%
plt.imsave("samples/img.png", t2i(image))
plt.imsave("samples/y_img.png", t2i(image_y), cmap="gray")
plt.imsave("samples/uv_img.png", t2i(showed_uv))
plt.imsave("samples/uv_distorted.png", t2i(distorted_uv))
plt.imsave("samples/distorted_img.png", t2i(distorted_image_rgb))
plt.imsave("samples/zoomed_img.png", t2i(z_img))
plt.imsave("samples/zoomed_distorted_img.png", t2i(z_distorted_image))
plt.imsave("samples/zoomed_uv.png", t2i(z_uv), dpi=300)
plt.imsave("samples/zoomed_distorted_uv.png", t2i(z_distorted_uv), dpi=300)
# %%

fig_flow = Flow(50, 50, parameterization=lambda x: torch.tanh(x) / 3)
src.utils.visualize_flow(fig_flow)
# %%
