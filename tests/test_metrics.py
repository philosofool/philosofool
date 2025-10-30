import torch
from philosofool.torch.metrics import Accuracy
import pytest

def test_accuracy__binary():
    pred = torch.tensor([.9, .8, .2, .1])
    true = torch.tensor([1, 0, 1, 0])
    accuracy = Accuracy('binary')

    assert accuracy.compute() == 0.0, "When count is zero, accuracy should be zero."

    accuracy.update(pred, torch.tensor([1, 0, 1, 0]))
    accuracy.update(pred, torch.tensor([1, 1, 1, 0]))
    result = accuracy.compute()
    assert result == 5/8

@pytest.mark.parametrize('pred', [
        torch.tensor([
            [.45, .35, .2],
            [.8, .1, .1],
            [.4, .49, .11],
            [.3, .4, .3]
        ]),
        torch.tensor([
            [.9, .1],
            [.8, .2],
            [.2, .8],
            [.3, .7]
        ])
])
def test_accuracy__multiclass(pred):
    true = torch.tensor([0, 0, 1, 0])
    accuracy = Accuracy('multiclass')
    accuracy.update(pred, true)
    assert accuracy.compute() == .75, \
        "The returned value should be a number corresponding to accuracy."

    accuracy.reset()
    accuracy.compute() == .0, "After reseting, accuracy should be zero again."
