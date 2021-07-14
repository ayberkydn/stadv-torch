#%%
import os
import torch, torchvision
import src
from torch.utils.data import DataLoader
import tqdm
import matplotlib.pyplot as plt
from kornia import tensor_to_image as t2i
from pathlib import Path

#%%
dataset = src.utils.NIPS2017TargetedDataset("data/nips2017_targeted")

net = torchvision.models.inception_v3().cuda()
net.load_state_dict(torch.load("models/inception_v3_google-1a9a5a14.pth"))
net.eval()

experiment_name = "nips2017_colorspace_stadv"
results_path = Path("./") / "results" / experiment_name
results_path.mkdir(parents=True, exist_ok=True)
#%%

fooled_images = []
not_fooled_images = []
for data in dataset:
    img = data["image"].cuda().unsqueeze(0)
    true_class = data["true_class"]
    target_class = data["target_class"]

    flow_layer = src.layers.Flow(299, 299, parameterization=None).to("cuda")
    optimizer = torch.optim.Adam(flow_layer.parameters(), lr=0.1)

    K = 0
    for n in tqdm.tqdm(range(500)):
        flowed_img = src.utils.flow_uv(img, flow_layer)
        out = net(flowed_img * 2 - 1)[0]
        loss = src.losses.adv_loss(out, target_class, K)
        if loss.item() == K:
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    fooled = loss.item() == K
    instance_save_path = results_path / data["image_name"]
    instance_save_path.mkdir(exist_ok=True)
    # instance_logs = []
    # if loss.item() == 0:
    #     fooled = True
    #     fooled_images.append(data["image_name"])
    #     instance_logs.append(f"Fooled within {n} iterations")
    # else:
    #     fooled = False
    #     instance_logs.append(f"Not fooled")
    #     not_fooled_images.append(data["image_name"])

    # with open(os.path.join(instance_save_path, "logs.txt"), "w") as logfile:
    #     logfile.writelines(instance_logs)

    plt.imsave(instance_save_path / "benign_img.png", t2i(img))

    if fooled:
        flowed_img = torch.clamp(src.utils.flow_uv(img, flow_layer), 0, 1)
        plt.imsave(instance_save_path / "adversarial_img.png", t2i(flowed_img))

# %%
