import torch
from einops import rearrange, reduce, repeat


def apply_flow(img_batch, flow_batch):

    assert img_batch.device == flow_batch.device
    assert len(img_batch.shape) == 4
    assert flow_batch.shape[-1] == 2

    assert flow_batch.shape[0] == img_batch.shape[0]
    assert flow_batch.shape[1:-1] == img_batch.shape[-2:]

    device = img_batch.device
    H, W = img_batch.shape[-2:]
    BATCH_SIZE = img_batch.shape[0]

    batched_grid = repeat(
        torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W))),
        "c h w -> b h w c",
        b=BATCH_SIZE,
    ).to(device)

    sampling_grid = batched_grid + flow_batch

    added = repeat(
        torch.tensor(
            [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long, device=device,
        ),
        "ind c -> b h w ind c",
        b=BATCH_SIZE,
        h=H,
        w=W,
    )

    sampled_pixel_indices = (
        repeat(torch.floor(sampling_grid).long(), "b h w c -> b h w 4 c",) + added
    )

    sampled_pixel_distances = torch.abs(
        sampled_pixel_indices - repeat(sampling_grid, "b h w c -> b h w c2 c", c2=4)
    )
    sampled_pixel_distances_h = sampled_pixel_distances[:, :, :, :, 0]
    sampled_pixel_distances_w = sampled_pixel_distances[:, :, :, :, 1]
    sampled_pixel_weights = (1 - sampled_pixel_distances_h) * (
        1 - sampled_pixel_distances_w
    )

    sampled_pixel_indices_y = sampled_pixel_indices[:, :, :, :, 0]
    sampled_pixel_indices_x = sampled_pixel_indices[:, :, :, :, 1]

    sampled_pixel_indices_y = torch.clamp(sampled_pixel_indices_y, 0, H - 1)
    sampled_pixel_indices_x = torch.clamp(sampled_pixel_indices_x, 0, W - 1)

    sampled_pixel_indices_reduced = (
        sampled_pixel_indices_y * W + sampled_pixel_indices_x
    )
    sampled_pixel_indices_reduced_flat = repeat(
        sampled_pixel_indices_reduced, "b h w four -> b c (h w four)", c=3
    )
    # sampled_pixel_indices_reduced_flat = repeat(sampled_pixel_indices_reduced, 'b h w four -> b c (h w four)', c = 1)

    img_batch_flat = rearrange(img_batch, "b c h w -> b c (h w)")

    sampled_pixels_flat = torch.gather(
        input=img_batch_flat, index=sampled_pixel_indices_reduced_flat, dim=-1
    )
    sampled_pixels = rearrange(
        sampled_pixels_flat, "b c (h w four) -> b c h w four", h=H, w=W, four=4
    )

    sampled_pixels_weighted = sampled_pixels * repeat(
        sampled_pixel_weights, "b h w four -> b c h w four", c=3
    )
    sampled_pixels_weighted_sum = reduce(
        sampled_pixels_weighted, "b c h w four -> b c h w", reduction="sum"
    )

    return sampled_pixels_weighted_sum
