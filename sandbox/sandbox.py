# %%
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms.functional as ttf
import src
from src.flow import Flow
from src.utils import t_imshow
import importlib

for n in [
    25,
]:  # 19, 10, 7, 5, 4, 2, 1, 0]:

    for mode in ["rgb", "ycbcr", "lab"]:

        data = src.utils.NIPS2017TargetedDataset("./data")[n]
        img = data["image"].unsqueeze(0).cuda()
        net = src.utils.load_net()
        attacker = src.attacker.SpatialAttacker(
            img, net, batch_size=1, mode=mode, is_restricted=False
        )

        target_class = data["target_class"]
        target_class_name = data["target_class_name"]
        attacker.attack_targeted(target_class, max_iters=1000, K=10, lr=0.1)

        summary = attacker.get_attack_summary()
        logits_probs = attacker.get_logits_probs()

        target_prob = summary.target_prob
        target_class_name = data["target_class_name"]
        # t_imshow(summary["benign_tensor"])
        # t_imshow(summary["adversarial_tensor"])

        # plt.imshow(attacker.get_benign_np())
        # plt.title(f"Benign image")
        # plt.axis("off")
        # plt.savefig(
        #     f"figures/examples/benign_{n}.png", bbox_inches="tight", pad_inches=0
        # )
        # plt.show()

        # plt.imshow(attacker.get_adversarial_np())
        # plt.axis("off")
        # plt.title(f'"{target_class_name}": p={target_prob:0.3f}')
        # plt.savefig(
        #     f"figures/examples/example_{mode}_{n}.png",
        #     bbox_inches="tight",
        #     pad_inches=0,
        # )
        # plt.show()

        print(summary)
        print("---")

    # # %%
    # a = Flow(299, 299, 1, 0, param=lambda x: 50 * torch.tanh(x + 50)).cuda()
    # t_imshow(a(img))


# %%
