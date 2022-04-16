import torch, torchvision
from piqa.tv import tv


def flow_loss(flow_layer):
    return tv(flow_layer.get_applied_flow())


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
