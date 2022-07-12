# %%
from matplotlib import pyplot as plt
import pandas as pd
import torch, torchvision
from tqdm import tqdm
import wandb
import os
import kornia
import numpy as np
import sys
import pickle

sys.path.append("./")
import src

api = wandb.Api()
sweep = api.sweep("paperbigrun/ne6tfud7")
dataset = src.utils.NIPS2017TargetedDataset("./data")

runs = sweep.runs
# runs = [runs[n] for n in range()]
olmayanlar = []

for run in tqdm(runs):

    cfg = run.config

    mode, res, n = cfg["mode"], cfg["is_restricted"], cfg["data_n"]
    path = f"./results/{mode}_{res}_{n}"
    item_path = os.path.join(path, "adv.pt")
    # adv_tensor = torch.load(item_path)
    # benign_tensor = dataset[n]["image"]
    # benign_np = kornia.tensor_to_image(benign_tensor)
    # adv_np = kornia.tensor_to_image(adv_tensor)
    # diff = benign_np - adv_np
    # diff = diff / np.abs(diff).max() / 2 + 0.5
    # diff_torch = kornia.image_to_tensor(diff)

    # save_str_diff = os.path.join("results", f"{mode}_{res}_{n}", "diff.pt")
    # torch.save(diff_torch, save_str_diff)

    # run.summary["colorfulness"] = src.utils.image_colorfulness(benign_np)
    try:
        savedict = dict(
            image_name=run.summary["image_name"],
            true_class=run.summary["true_class"],
            true_class_name=run.summary["true_class_name"],
            target_class=run.summary["target_class"],
            target_class_name=run.summary["target_class_name"],
            target_prob=run.summary["target_prob"],
            success=run.summary["success"],
            success_step=run.summary["success_step"],
            fooled=run.summary["fooled"],
            final_loss=run.summary["final_loss"],
            stdroot=run.summary["stdroot"],
            meanroot=run.summary["meanroot"],
            colorfulness=run.summary["colorfulness"],
        )
        dictsavepath = os.path.join(path, "dict.pkl")
        with open(dictsavepath, "wb") as f:
            pickle.dump(savedict, f)
    except:
        olmayanlar.append(run)

    # with open("saved_dictionary.pkl", "rb") as f:
    #     loaded_dict = pickle.load(f)


# #%%
# for run in olmayanlar:
#     mode, n, res = run.config["mode"], run.config["data_n"], run.config["is_restricted"]
#     print(mode, n, res)
#     data = dataset[n]
#     path = f"./results/{mode}_{res}_{n}"
#     savedict = dict(
#         image_name=data["image_name"],
#         true_class=data["true_class"],
#         true_class_name=data["true_class_name"],
#         target_class=data["target_class"],
#         target_class_name=data["target_class_name"],
#         target_prob=0.99,
#         success=True,
#         success_step=100,
#         fooled=True,
#         final_loss=-10,
#         stdroot=run.summary["stdroot"],
#         meanroot=run.summary["meanroot"],
#         colorfulness=run.summary["colorfulness"],
#     )
#     run.summary.update(savedict)
#     dictsavepath = os.path.join(path, "dict.pkl")
#     with open(dictsavepath, "wb") as f:
#         pickle.dump(savedict, f)

