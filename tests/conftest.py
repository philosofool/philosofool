import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


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