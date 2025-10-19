import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from philosofool.torch.nn_loop import TrainingLoop


@pytest.fixture
def dataset() -> TensorDataset:
    # 1 batch, 2 rows, three columns
    data = torch.tensor(
        [[1., 0., 0], [0., 1., .0]]
    )
    labels = torch.tensor(
        [[1., 0.], [0., 1.]]
    )
    return TensorDataset(data, labels)


@pytest.fixture
def data_loader(dataset) -> DataLoader:
    data_loader = DataLoader(dataset, batch_size=2)
    return data_loader

@pytest.fixture
def training_loop() -> TrainingLoop:
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=.5)
    loss = nn.CrossEntropyLoss()

    training_loop = TrainingLoop(model, optimizer, loss)
    return training_loop


class SimpleModel(nn.Module):
    """A linear model that takes three inputs and returns two outputs."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x):
        logits = self.linear(x)
        return logits