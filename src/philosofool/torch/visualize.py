from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def show_image(img: torch.Tensor):
    """Show image implied by input tensor. The input is assumed to be normalized."""
    img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()