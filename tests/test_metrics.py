import torch
from philosofool.torch.metrics import Accuracy

def test_accuracy__binary():
    pred = torch.tensor([.9, .8, .2, .1])
    true = torch.tensor([1, 0, 1, 0])
    accuracy = Accuracy('binary')

    assert accuracy.compute() == 0.0, "When count is zero, accuracy should be zero."

    accuracy.update(pred, torch.tensor([1, 0, 1, 0]))
    accuracy.update(pred, torch.tensor([1, 1, 1, 0]))
    result = accuracy.compute()
    assert result == 5/8

def test_accuracy__multiclass():
    pred = torch.tensor([
        [.9, .1],
        [.8, .2],
        [.2, .8],
        [.3, .7]
    ])
    true = torch.tensor([0, 0, 1, 0])
    accuracy = Accuracy('multiclass')
    accuracy.update(pred, true)
    assert accuracy.compute() == .75

    accuracy.reset()
    accuracy.compute() == .0, "After reseting, accuracy should be zero again."
