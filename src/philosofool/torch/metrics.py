from typing import Protocol
from numpy.typing import ArrayLike
import torch
import numpy as np
from typing import Literal

class Metric(Protocol):
    """Describes a metric computed by an operation on model predictions."""
    def update(self, y_hat: ArrayLike, y_true: ArrayLike) -> None:
        """Accumulates the predicted and correct values."""
        ...

    def compute(self) -> float | ArrayLike:
        """Compute the values of the metric from the values update accumulates."""
        ...

    def reset(self) -> None:
        """Set the values for the accumulators to their base state."""
        ...


class Accuracy:
    """Track accuracy of predictions.

    task:
        'multiclass' or 'binary', this determines the expected inputs to update.
    threshold:
        The threshold for a positive classification as the class labeled one;
        used only when the task is binary. The default is .5.
    """
    def __init__(self, task: Literal['binary', 'multiclass'], threshold: float = .5):
        self.count = 0
        self.correct = 0
        self.threshold = None if task == 'multiclass' else threshold
        self.task = task

    def update(self, y_hat: ArrayLike, y_true: ArrayLike):
        """Accumulate the count of correct predictions.

        This method is typically called on each batch of predictions made
        by a model. The compute method is used to return the accuracy across
        all batches.

        Parameters
        ----------

        y_hat
            The probaility assigned to the classes.
            If task is multiclass, this is expected to be n_samples x n_classes
            shape matrx.
        y_true
            The correct class labels. Expected to be a 1-dim array-like of values,
            corresponding to the class labels.
        """
        y_hat = np.asarray(y_hat)
        y_true = np.asarray(y_true)
        self.correct += self._compute_correct(y_hat, y_true)
        self.count += y_true.size

    def _compute_correct(self, y_hat, y_true):
        """Handle task type and return corect predictions."""
        if self.task == 'binary':
            correct = (y_hat > self.threshold) == y_true
        else:
            correct = np.argmax(y_hat, axis=1) == y_true
        return np.sum(correct)

    def compute(self):
        """Return the fraction of correct predictions based on the threshold.

        The values to compute are passed in batches via the update method.
        This method returns the current result of the computation over the batch
        results.
        """
        return self.correct / self.count if self.count else 0.0

    def reset(self):
        """Reset counts and correct values accumulated via update.

        This method is typically called at the end of each epoch to allow
        new batches to reflect the results of the last epoch's updates.
        """
        self.count = 0.
        self.correct = 0.