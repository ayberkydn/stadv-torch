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
    """
    adv_logits: [N]
    target_class: int
    kappa: non-positive float
    """
    top_two_logits, top_two_classes = torch.topk(adv_logits, 2)
    target_class_logit = adv_logits[target_class]

    if top_two_classes[0] == target_class:
        nontarget_max = top_two_logits[1]
    else:
        nontarget_max = top_two_logits[0]

    loss = torch.maximum(nontarget_max - target_class_logit, torch.tensor(kappa))
    return loss
