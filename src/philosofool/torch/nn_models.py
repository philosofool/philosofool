import torch
from torch import nn
from torch.nn import functional as F


class NeuralNetwork(torch.nn.Module):
    """A feedforward network for image classification."""

    def __init__(self, input_dims: int, out_dims: int, width: int, n_hidden_layers: int):
        super().__init__()
        # we could also use nn.Sequential. This is uglier, but requires
        # thinking about the steps more clearly.
        self.flatten = nn.Flatten()
        self.h1 = nn.Sequential(nn.Linear(input_dims, width), nn.ReLU())
        linear_relu_stack = [nn.Sequential(nn.Linear(width, width), nn.ReLU()) for _ in range(n_hidden_layers)]
        self.linear_relu_stack = nn.Sequential(*linear_relu_stack)
        self.output_layer = nn.Linear(width, out_dims)

    def forward(self, x):
        x = self.flatten(x)

        x = self.h1(x)
        x = self.linear_relu_stack(x)
        logits = self.output_layer(x)
        return logits

class ConvModel(nn.Module):
    """A convolution neural network."""

    def __init__(self, input_dims: tuple[int, int], in_channels=1, out_dims: int = 1):
        super().__init__()
        convolutions = [
            nn.Conv2d(in_channels, 6, 7),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 11, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ]

        h, w, channels = compute_convolution_dims(
            input_dims,
            *[conv for conv in convolutions if isinstance(conv, (nn.Conv2d, nn.MaxPool2d))])
        assert channels is not None
        flat_dims = h * w * channels

        self.convolutions = nn.Sequential(*convolutions)
        self.h1 = nn.Sequential(nn.Linear(flat_dims, 128), nn.ReLU())
        self.h2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.output = nn.Linear(64, out_dims)

    def forward(self, x):
        x = self.convolutions(x)
        x = nn.Flatten()(x)
        x = nn.Dropout(.1)(x)
        x = self.h1(x)
        x = nn.Dropout(.1)(x)
        x = self.h2(x)
        x = nn.Dropout(.1)(x)
        x = self.output(x)
        return x

class ResidualNetwork(nn.Module):
    def __init__(self, input_dims: tuple[int, int], in_channels: int, out_features: int):
        super().__init__()
        stride = 2 if max(input_dims) > 128 else 1
        self.residual_layer = nn.Sequential(
            nn.Conv2d(in_channels, 7, 7, stride=stride),
            ResidualBlock(7, 12, 3, stride=1),
            ResidualBlock(12, 12, 5, 1),
            nn.MaxPool2d(2, 2),
            ResidualBlock(12, 17, 5, 1),
            nn.MaxPool2d(2, 2)
        )

        # FIXME: this will break on a too-small input, e.g. 28x28. Y
        height, width, channels = compute_convolution_dims(
            input_dims, self.residual_layer
        )

        self.fc1 = nn.Linear(height * width * channels, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, out_features)

    def forward(self, x):
        x = self.residual_layer(x)

        x = nn.Flatten()(x)
        x = nn.Dropout(.3)(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x


def _to_2d_tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2, f"Expected length 2 tuple. Got {x}."
        return x
    return (x, x)


def conv_dims_1d(
        input_dims: tuple[int, int],
        kernel_size: int | tuple[int, int],
        padding: int | tuple[int, int],
        stride: int | tuple[int, int]
) -> tuple[int, int]:
    kernel_size = _to_2d_tuple(kernel_size)
    padding = _to_2d_tuple(padding)
    stride = _to_2d_tuple(stride)
    out = tuple((input_dims[i] - kernel_size[i] + 2 * padding[i]) // stride[i] + 1 for i in range(2))
    return out    # pyright: ignore [reportReturnType]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, n_steps: int = 2):
        super().__init__()
        # self._input_dims = input_dims
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_steps = n_steps

        block_steps = []
        for i in range(n_steps):
            padding = kernel_size // 2
            if i == 0:
                conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
            else:
                conv_layer = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=padding)
            block_steps.append(conv_layer)
            block_steps.append(nn.BatchNorm2d(out_channels))
            # don't apply the activation for the final convolution (which is activated after the skip connection.)
            if i != n_steps - 1:
                block_steps.append(nn.ReLU())
        self.residual_layer = nn.Sequential(*block_steps)

        if in_channels != out_channels or (stride != (1, 1) and stride != 1):
            self.normalize_inputs = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, stride=stride))
        else:
            self.normalize_inputs = nn.BatchNorm2d(in_channels)

    def output_size(self, input_dims: tuple[int, int]):
        out_dims = input_dims
        for i in range(self.n_steps):
            padding = self.kernel_size // 2
            out_dims = conv_dims_1d(input_dims, self.kernel_size, padding, self.stride)
        return out_dims

    def forward(self, x):
        x_ = self.normalize_inputs(x)
        x = self.residual_layer(x)
        x = x + x_
        x = F.relu(x)
        return x


class Generator(nn.Module):
    def __init__(self, input_size: int, features_size: int, dropout=.1):
        """
        Parameters
        ----------
        input_size:
            The size of the generator latent spaces, i.e., the input vector.
        features_size
            A parameter controlling the number of channels in each convolution.
            Each layers has some multiple of this number of output channels.
            A way to control generator capacity.
        dropout:
            A value used for dropout layers (NOT IMPLEMENTED.)
        """
        super().__init__()
        self.input_size = input_size
        self.features_size = features_size
        features_size = features_size

        def apply_batch_norm_and_relu(layer: nn.ConvTranspose2d) -> nn.Sequential:
            """Apply batch norm, leakyReLU and optional dropout after layer."""
            steps = [
                layer,
                nn.BatchNorm2d(layer.out_channels),
                nn.LeakyReLU()
            ]
            if dropout:
                steps.append(nn.Dropout(dropout))
            return nn.Sequential(*steps)

        self.network = nn.Sequential(
            *[
                apply_batch_norm_and_relu(layer) for layer in (
                    nn.ConvTranspose2d(input_size, features_size * 8, 4, 2, 1),
                    nn.ConvTranspose2d(features_size * 8, features_size * 4, 4, 2, 0),
                    nn.ConvTranspose2d(features_size * 4, features_size * 4, 4, 2, 0),
                    nn.ConvTranspose2d(features_size * 4, features_size * 4, 4, 2, 0),
                    nn.ConvTranspose2d(features_size * 4, features_size * 4, 4, 2, 0),
                    nn.ConvTranspose2d(features_size * 4, features_size * 4, 4, 1, 1),
                    nn.ConvTranspose2d(features_size * 4, features_size * 2, 4, 1, 0)
            )],
            nn.Conv2d(features_size * 2, 3, 1, 1, 0),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)

def is_window_function(layer):
    return hasattr(layer, 'padding') and hasattr(layer, 'stride') and hasattr(layer, 'kernel_size')


def compute_convolution_dims(input_dims, *layers) -> tuple[int, int, int | None]:
    h, w = input_dims
    channels = None
    for layer in layers:
        if isinstance(layer, nn.Sequential):
            h, w, channels = compute_convolution_dims((h, w), *layer)
            continue
        if isinstance(layer, ResidualBlock):
            h, w = layer.output_size((h, w))
            channels = layer.out_channels
            continue
        if not is_window_function(layer):
            continue
        h, w = conv_dims_1d((h, w), layer.kernel_size, layer.padding, layer.stride)
        channels = getattr(layer, 'out_channels', channels)
    return h, w, channels


class Discriminator(nn.Module):
    def __init__(self, feature_size: int = 10, dropout=.0, expected_input_size: tuple[int, int] = (64, 64)):
        super().__init__()
        self.dropout = dropout

        def apply_batch_norm_and_relu(layer: nn.Conv2d | nn.MaxPool2d) -> nn.Module:
            """Apply batch norm, leakyReLU and optional dropout after layer."""

            if isinstance(layer, nn.MaxPool2d):
                return layer

            steps = [
                layer,
                # nn.BatchNorm2d(layer.out_channels),
                nn.ReLU()
            ]
            if dropout:
                steps.append(nn.Dropout(dropout))
            return nn.Sequential(*steps)

        self.steps = [
            apply_batch_norm_and_relu(layer) for layer in (
                nn.Conv2d(3, feature_size, 3, stride=1),
                nn.Conv2d(feature_size, feature_size * 2, 3, stride=1),
                nn.Conv2d(feature_size * 2, feature_size * 2, 3, stride=1),
                nn.Conv2d(feature_size * 2, feature_size * 2, 3, stride=1),
                nn.Conv2d(feature_size * 2, feature_size * 4, 5, stride=1),
                nn.MaxPool2d(2),
                nn.Conv2d(feature_size * 4, feature_size * 4, 5, stride=1),
                nn.Conv2d(feature_size * 4, feature_size * 4, 5, stride=1),
                nn.MaxPool2d(2),
                nn.Conv2d(feature_size * 4, feature_size * 8, 7, stride=1),
            )
        ]
        h, w, c = compute_convolution_dims(expected_input_size, *self.steps)
        linear_width = 32
        self.network = nn.Sequential(
           *self.steps,
            nn.Flatten(),
            nn.Linear(h * w * c, linear_width),
            nn.ReLU(),
            nn.Linear(linear_width, 1)
        )

    def forward(self, x):
        # x = nn.Dropout(.3)(x)
        return self.network(x)

if __name__ == '__main__':
    dis = Discriminator(60)
    print(compute_convolution_dims((64, 64), *dis.steps))
