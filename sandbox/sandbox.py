# %%
from matplotlib import pyplot as plt
import torch, torchvision
import wandb
import src
import os
import kornia
import numpy as np
import src.utils

api = wandb.Api()
dataset = src.utils.NIPS2017TargetedDataset("./data")

print(run.config)
hist = run.history()
summary = run.summary
arts = run.logged_artifacts()
art = arts[0]

if not os.path.exists(art.file()):
    print(f"Downloading {art.file()}")
    art.download()
mode, res, n = run.config["mode"], run.config["is_restricted"], run.config["data_n"]
item_path = f"./artifacts/{mode}_{res}_{n}:v0/adv.pt"

item_n = run.config["data_n"]
adv_tensor = torch.load(item_path)
benign_tensor = dataset[item_n]["image"]
benign_np = kornia.tensor_to_image(benign_tensor)
adv_np = kornia.tensor_to_image(adv_tensor)
diff = benign_np - adv_np
diff = diff / np.abs(diff).max() / 2 + 0.5
# plt.imshow(benign_np)
# plt.show()
# plt.imshow(adv_np)
# plt.show()
# plt.imshow(diff)
# plt.show()

run.summary["colorfulness"] = src.utils.image_colorfulness(benign_np)
# run.update()
#%%
# for file in run.files():
#     print(file)
#     try:
#         file.download()
#     except:
#         pass
