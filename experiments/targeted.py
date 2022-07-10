#%%
import enum
import itertools
import os
import numpy as np
import torch, torchvision
from torch.utils.data import DataLoader
import tqdm
import matplotlib.pyplot as plt
from kornia import image_to_tensor, tensor_to_image as t2i
from pathlib import Path
import wandb
import sys
import os

sys.path.append("./")
from src.attacker import SpatialAttacker
import src


is_restricteds = [True, False]
modes = ["rgb", "ycbcr", "lab"]
cfg = dict(
    mode="rgb",
    datanumber=124,
    is_restricted=True,
    kappa=10,
    batch_size=4,
    max_iters=1000,
)

# hparam_defaults = dict(mode="rgb", is_restricted=False, kappa=10,)


with wandb.init(project="stadv") as run:
    run.config.setdefaults(cfg)
    mode = run.config["mode"]
    batch_size = run.config["batch_size"]
    is_restricted = run.config["is_restricted"]
    max_iters = run.config["max_iters"]
    kappa = run.config["kappa"]
    net = src.utils.load_net()
    dataset = src.utils.NIPS2017TargetedDataset("./data", head=10)

    art_name = f"{mode}_{is_restricted}"
    save_dir = os.path.join("results", art_name)
    os.makedirs(save_dir, exist_ok=True)

    for n, data in enumerate(dataset):

        true_class = data["true_class"]
        target_class = data["target_class"]
        img = data["image"].cuda().unsqueeze(0)
        attacker = SpatialAttacker(
            img, net, mode=mode, batch_size=batch_size, is_restricted=is_restricted
        )
        success = attacker.attack_targeted(target_class, max_iters=max_iters, K=kappa)
        summary = attacker.get_attack_summary()
        benign_img = attacker.get_benign_img("tensor")
        adversarial_img = attacker.get_adversarial_img("tensor")
        log_dict = dict(
            true_class=true_class,
            target_class=target_class,
            benign_img=wandb.Image(benign_img),
            adversarial_img=wandb.Image(adversarial_img),
            final_loss=summary.final_loss,
            fooled=summary.final_loss <= 0,
            success=summary.final_loss <= kappa,
            success_step=summary.success_step,
        )
        run.log(log_dict)

        save_path = os.path.join(save_dir, f"{n}.pt")
        torch.save(adversarial_img, save_path)

    artifact = wandb.Artifact(art_name, type="dataset")
    artifact.add_dir(save_dir)
    run.log_artifact(artifact)


# %%

# %%
