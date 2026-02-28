import torch
import torch.nn as nn


class FeasibilityLoss(nn.Module):
    """Penalizes (x, y) predictions falling outside the U-maze corridor.

    Uses differentiable projection onto skeleton segments to allow gradient flow.
    """

    def __init__(self, skeleton_segments, corridor_half_width):
        super().__init__()
        self.register_buffer('segments', torch.tensor(skeleton_segments, dtype=torch.float32))
        self.corridor_half_width = corridor_half_width

    def forward(self, xy_pred):
        px, py = xy_pred[:, 0], xy_pred[:, 1]
        distances = []
        for i in range(self.segments.shape[0]):
            x1, y1, x2, y2 = self.segments[i]
            dx, dy = x2 - x1, y2 - y1
            seg_len_sq = dx**2 + dy**2
            t = ((px - x1) * dx + (py - y1) * dy) / (seg_len_sq + 1e-8)
            t = t.clamp(0.0, 1.0)
            proj_x, proj_y = x1 + t * dx, y1 + t * dy
            dist = torch.sqrt((px - proj_x)**2 + (py - proj_y)**2 + 1e-8)
            distances.append(dist)
        distances = torch.stack(distances, dim=1)
        min_dist = distances.min(dim=1).values
        return torch.relu(min_dist - self.corridor_half_width).pow(2).mean()
