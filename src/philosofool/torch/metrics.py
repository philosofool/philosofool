import torch
import numpy as np
from typing import Literal

class Metric:
    def update(self, *args, **kwargs):
        ...

    def compute(self):
        ...


class Accuracy:
    def __init__(self, task: Literal['binary', 'multiclass'], threshold: float = .5):
        self.count = 0
        self.correct = 0
        self.threshold = None if task == 'multiclass' else threshold
        self.task = task

    def update(self, y_hat, y_true):
        y_hat = np.asarray(y_hat)
        y_true = np.asarray(y_true)
        self.correct += self._compute_correct(y_hat, y_true)
        self.count += y_true.size

    def _compute_correct(self, y_hat, y_true):
        if self.task == 'binary':
            correct = (y_hat > self.threshold) == y_true
        else:
            correct = np.argmax(y_hat) == y_true
        return np.sum(correct)


    def compute(self):
        return self.correct / self.count if self.count else 0.0

    def reset(self):
        self.count = 0.
        self.correct = 0.