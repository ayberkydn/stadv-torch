#%%
import os
import torch, torchvision
import src
from torch.utils.data import DataLoader
import tqdm
import matplotlib.pyplot as plt
from kornia import tensor_to_image as t2i
from pathlib import Path

experiment_name = "oguza_atilacak"
#%%
dataset = src.utils.NIPS2017TargetedDataset("data/nips2017_targeted")

net = torchvision.models.inception_v3().cuda()
net.load_state_dict(torch.load("models/inception_v3_google-1a9a5a14.pth"))
net.eval()

results_path = Path("./") / "results" / experiment_name
results_path.mkdir(parents=True, exist_ok=True)
#%%

fooled_images = []
not_fooled_images = []
datasubset = [dataset[n] for n in range(10)]
for data in datasubset:
    img = data["image"].cuda().unsqueeze(0)
    true_class = data["true_class"]
    target_class = data["target_class"]

    flow_layer = src.layers.Flow(299, 299, param=None).to("cuda")
    optimizer = torch.optim.Adam(flow_layer.parameters(), lr=0.1)

    K = -2
    tau = 0.002
    adv_losses = []
    flow_losses = []
    for n in tqdm.tqdm(range(200)):
        # flowed_img = src.utils.flow_uv(img, flow_layer)
        flowed_img = flow_layer(img)
        out = net(flowed_img * 2 - 1)[0]
        adv_loss = src.losses.adv_loss(out, target_class, K)
        flow_loss = src.losses.flow_loss(flow_layer)
        loss = adv_loss + flow_loss * tau

        optimizer.zero_grad()
        loss.backward()
        adv_losses.append(adv_loss.item())
        flow_losses.append(flow_loss.item())
        optimizer.step()
    plt.plot(adv_losses)
    plt.show()
    plt.plot(flow_losses)
    plt.show()

    fooled = adv_loss.item() <= 0
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
        flowed_img = torch.clamp(flow_layer(img), 0, 1)
        plt.imsave(instance_save_path / "adversarial_img.png", t2i(flowed_img))

# %%

# %%
