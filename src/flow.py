# %%
import torch


class Flow(torch.nn.Module):
    def __init__(
        self, height, width, batch_size=1, init_std=1, param=None,
    ):
        # parameterization is the function to apply
        # to self.flow_field before applying the flow

        super().__init__()

        self.H = height
        self.W = width
        self.batch_size = batch_size

        self.basegrid = torch.nn.Parameter(
            torch.cartesian_prod(
                torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W)
            )
            .view(self.H, self.W, 2)
            .unsqueeze(0)
            .repeat_interleave(self.batch_size, dim=0)
        )
        self.basegrid.requires_grad = False

        if param is None:
            self.parameterization = torch.nn.Identity()
        else:
            self.parameterization = param

        self._pre_flow_field = torch.nn.Parameter(
            torch.randn([self.batch_size, self.H, self.W, 2]) * init_std,
            requires_grad=True,
        )

    def _normalize_grid(self, in_grid):
        """
            Normalize x and y coords of in_grid into range -1, 1
            to keep torch.grid_sample happy
        """
        grid_x = in_grid[..., 0]
        grid_y = in_grid[..., 1]

        return torch.stack([grid_x * 2 / self.H, grid_y * 2 / self.W], dim=-1)

    def forward(self, x):
        assert len(x.shape) == 4  # Image does contain batch dim
        assert x.shape[0] == self._pre_flow_field.shape[0], "NOT SAME SHAPE!"

        grid = self.basegrid + \
            self._normalize_grid(self.get_applied_flow_pixels())

        return torch.nn.functional.grid_sample(
            x, grid, align_corners=True, padding_mode="reflection"
        ).mT

    def get_applied_flow_pixels(self):
        return self.parameterization(self._pre_flow_field)
