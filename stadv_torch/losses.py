import torch, torchvision


def flow_loss(flow_layer):
    applied_flow = flow_layer.get_applied_flow()
    padded_flow = torchvision.transforms.Pad(
        padding=[1, 1],
        padding_mode="reflect",
    )(applied_flow)

    right = padded_flow[:, 1:-1, 2:]
    left = padded_flow[:, 1:-1, :-2]
    down = padded_flow[:, 2:, 1:-1]
    up = padded_flow[:, :-2, 1:-1]

    d_right = torch.sqrt(torch.sum(torch.square(applied_flow - right), dim=0))
    d_left = torch.sqrt(torch.sum(torch.square(applied_flow - left), dim=0))
    d_down = torch.sqrt(torch.sum(torch.square(applied_flow - down), dim=0))
    d_up = torch.sqrt(torch.sum(torch.square(applied_flow - up), dim=0))

    return torch.sum(d_right + d_left + d_down + d_up)


def adv_loss(adv_logits, target_class, kappa):
    target_masked_logits = adv_logits.clone()
    target_masked_logits[target_class] = target_masked_logits.min() - 0.1
    nontarget_max = target_masked_logits.max()
    # nontarget_max will never be adv_logits[target_class]

    loss = torch.maximum(nontarget_max - adv_logits[target_class], torch.tensor(kappa))
    return loss
