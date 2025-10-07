from philosofool.torch.nn_models import ResidualBlock, compute_convolution_dims, conv_dims_1d, is_window_function
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import pytest

def test_compute_convolution_dims():
    layer = nn.Conv2d(1, 6, (2, 4))
    assert compute_convolution_dims((28, 28), layer) == (27, 25, 6)
    assert compute_convolution_dims((28, 28), nn.MaxPool2d(2, 2)) == (14, 14, None)
    result = compute_convolution_dims((28, 28), layer, nn.MaxPool2d(2, 2))
    assert result == (13, 12, 6), f"Got {result}"
    seq = nn.Sequential(layer, nn.MaxPool2d(2, 2))
    assert compute_convolution_dims((28, 28), seq) == (13, 12, 6)

    block = ResidualBlock(1, 3, 3, 1)
    result = compute_convolution_dims((28, 28), ResidualBlock(1, 3, 3, 1))
    assert result[:2] == block.output_size((28, 28))
    assert result[2] == block.out_channels

def test_is_window_function():
    assert is_window_function(nn.MaxPool2d(2, 2))
    assert is_window_function(nn.Conv2d(1, 1, 1))
    assert not is_window_function(nn.ReLU())
    assert not is_window_function(nn.Linear(12, 12))

@pytest.mark.parametrize('inputs, expected, msg',[
    ([(28, 28), 3, 1, 2], (14, 14), "Even valued input lenghts should compute correctly."),
    ([(29, 29), 3, 1, 2], (15, 15), "Odd valued inputs lengths should compute correcctly."),
    ([(12, 10), 3, 1, 1], (12, 10), "Dissmilar valued lengths should compute correctly."),
    ([(12, 12), (3, 5), 1, 1], (12, 10), "Dissimilar valued kernels should compute correctly.")
])
def test_convolution_dims(inputs, expected, msg):
    assert conv_dims_1d(*inputs) == expected, msg

class TestResidualBlock:
    def test_normalize_inputs(self):
        block = ResidualBlock(1, 5, kernel_size=3, stride=1)
        x = torch.tensor([[1., 2.], [-1, -1.5]]).reshape(1, 1, 2, 2)
        result = block.normalize_inputs(x)
        assert result.shape == (1, 5, 2, 2), f"Expected 5 channels in shape. Got {result.shape}"

    def test_output_size(self):
        block = ResidualBlock(1, 3, 3, 1)
        result = block.output_size((28, 28))
        assert result == (28, 28)
        block2 = ResidualBlock(1, 3, 3, 2)
        result = block2.output_size((28, 28))
        assert result == (14, 14)

    def test_forward(self):
        block = ResidualBlock(1, 3, 3, 1)
        x = torch.rand((6, 1, 3, 3))
        result = block.forward(x).detach().numpy()
        assert result.shape == (6, 3, 3, 3)

        block = ResidualBlock(1, 5, 5, 1)
        x = torch.rand((6, 1, 3, 3))
        result = block.forward(x).detach().numpy()
        assert result.shape == (6, 5, 3, 3)
