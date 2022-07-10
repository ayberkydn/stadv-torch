import torch
from . import flow
from . import utils
from tqdm import tqdm
from dataclasses import dataclass
import piqa
import PIL
from kornia import tensor_to_image as t2i
import numpy as np


class SpatialAttacker:
    def __init__(
        self,
        img: torch.Tensor,
        net: torch.nn.Module,
        batch_size: int = 1,
        mode: str = "rgb",
        is_restricted: bool = False,
    ):
        self.benign_tensor = img
        self.batch_size = batch_size
        if is_restricted:
            param = lambda x: torch.tanh(x)
        else:
            param = None
        self.flow = flow.Flow(
            img.shape[-2], img.shape[-1], batch_size=batch_size, param=param
        ).cuda()
        self.net = net
        self.attack_success = False
        self.mode = mode
        self.fooled = False
        self.confidently_fooled = False
        if mode == "rgb":
            self.flow_fn = utils.flow_rgb
        elif mode == "ycbcr":
            self.flow_fn = utils.flow_cbcr
        elif mode == "lab":
            self.flow_fn = utils.flow_ab
        else:
            raise NotImplementedError

    def attack_targeted(
        self,
        target_class,
        max_iters,
        K=0,
        lr=0.1,
        sch_patience=10,
        sch_factor=0.5,
        min_lr=0.001,
    ):
        assert K >= 0, "K must be nonnegative"
        self.target_class = target_class

        optim = torch.optim.Adam(self.flow.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=sch_patience, factor=sch_factor, verbose=False,
        )
        pbar = tqdm(range(max_iters))
        for n in pbar:
            flowed_img = self.flow_fn(self.benign_tensor, self.flow)
            out = self.net(flowed_img)

            losses, sum_loss = utils.adv_loss(out, target_class, K=K)
            self.final_loss = losses.min().item()
            pbar.set_description(f"Loss: {losses.min().item():0.3f}")

            self.best_attack_idx = torch.argmin(losses, dim=0)

            if losses.min() <= -K:
                self.success_step = n
                return True

            optim.zero_grad()
            sum_loss.backward()
            optim.step()
            scheduler.step(self.final_loss)
            if scheduler._last_lr[0] <= min_lr:
                break

        self.success_step = -1
        return False

    @torch.no_grad()
    def get_logits_probs(self):
        @dataclass
        class LogitsProbs:
            benign_logits: list
            adversarial_logits: list
            benign_probs: list
            adversarial_probs: list

        return LogitsProbs(
            benign_logits=self.get_benign_logits(),
            adversarial_logits=self.get_adversarial_logits(),
            benign_probs=self.get_benign_probs(),
            adversarial_probs=self.get_adversarial_probs(),
        )

    @torch.no_grad()
    def get_attack_summary(self):
        @dataclass
        class AttackSummary:
            target_prob: float
            success_step: int
            final_loss: float
            benign_img: np.ndarray
            adv_img: np.ndarray

        return AttackSummary(
            target_prob=self.get_adversarial_probs()[self.target_class].item(),
            final_loss=self.final_loss,
            success_step=self.success_step,
            benign_img=self.get_benign_img(),
            adv_img=self.get_adversarial_img(),
        )

    @torch.no_grad()
    def get_similarity(self):
        @dataclass
        class Similarity:
            lpips: float
            ssim: float
            ms_ssim: float

        lpips_fn = piqa.LPIPS().cuda()
        ssim_fn = piqa.SSIM().cuda()
        ms_ssim_fn = piqa.MS_SSIM().cuda()
        benign_tensor = self.get_benign_img("tensor").unsqueeze(0)
        adversarial_tensor = self.get_adversarial_img("tensor").unsqueeze(0)
        lpips = lpips_fn(benign_tensor, adversarial_tensor).item()
        ssim = ssim_fn(benign_tensor, adversarial_tensor).item()
        ms_ssim = ms_ssim_fn(benign_tensor, adversarial_tensor).item()
        return Similarity(lpips=lpips, ssim=ssim, ms_ssim=ms_ssim)

    @torch.no_grad()
    def get_benign_img(self, mode="np"):
        benign_tensor = self.benign_tensor
        if mode == "np":
            return t2i(benign_tensor)
        elif mode == "tensor":
            return benign_tensor[0]
        elif mode == "pillow":
            nparray = 255 * t2i(benign_tensor)
            return PIL.Image.fromarray(nparray.astype(np.uint8))

    @torch.no_grad()
    def get_adversarial_img(self, mode="np"):
        adv_tensor = self.flow_fn(self.benign_tensor, self.flow)[self.best_attack_idx]
        if mode == "np":
            return t2i(adv_tensor)
        elif mode == "tensor":
            return adv_tensor
        elif mode == "pillow":
            nparray = 255 * t2i(adv_tensor)
            return PIL.Image.fromarray(nparray.astype(np.uint8))

    @torch.no_grad()
    def get_benign_logits(self):
        return self.net(self.benign_tensor)[0]

    @torch.no_grad()
    def get_adversarial_logits(self):
        return self.net(self.flow_fn(self.benign_tensor, self.flow))[
            self.best_attack_idx
        ]

    @torch.no_grad()
    def get_benign_probs(self):
        return torch.nn.functional.softmax(self.get_benign_logits(), dim=-1)

    @torch.no_grad()
    def get_adversarial_probs(self):
        return torch.nn.functional.softmax(self.get_adversarial_logits(), dim=-1)


# %%
