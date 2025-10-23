from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterable
import json
import os
import numpy as np
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from philosofool.torch.visualize import show_image
from philosofool.torch.metrics import Metric

if TYPE_CHECKING:
    from philosofool.torch.nn_loop import GANLoop, TrainingLoop


class EarlyStoppingCallabck:
    def __init__(self, patience: int, monitor: str = 'test_loss'):
        self.patience = patience
        if monitor not in {'val_loss', 'test_loss'}:
            raise NotImplementedError("EarlyStopping currently only suppots stopping on validation loss.")
        self.monitor = monitor
        self._loops_since_improvement = 0
        self._best = None

    def on_batch_end(self, loop, batch, loss, **kwargs):
        ...

    def on_epoch_end(self, loop, test_loss, **kwargs):
        if self.monitor in {'test_loss', 'val_loss'}:
            return self._determine_improvement(loop, test_loss)
        raise NotImplementedError("EarlyStopping is only configured to monitor test_loss or val_loss.")

    def _determine_improvement(self, loop, target_metric):
        if self._best is None:
            self._best = target_metric
            return
        if target_metric < self._best:
            self._best = target_metric
            self._loops_since_improvement = 0
            return
        self._loops_since_improvement += 1
        if self._loops_since_improvement >= self.patience:
            loop.publish(f"{loop.name}_control", 'end_fit')


class EndOnBatchCallback:
    """After a number of batches, proceed to the next epoch."""

    def __init__(self, last_batch: int):
        self.last_batch = last_batch

    def on_batch_end(self, loop, batch: int, **kwargs):
        loop.publish(f"{loop.name}_control", 'end_epoch')


class SnapshotCallback:
    """Collect snapshots on an interval.

    With generative models, this collects the outputs on batches and epochs
    to assess the quality of generated images.
    """

    def __init__(self, n_images: int, interval: int):
        self.interval = interval
        self.n_images = n_images
        self.snapshots = []
        self._random_inputs = {}

    def on_batch_end(self, loop: GANLoop, batch, **kwargs):
        if batch % self.interval != 0:
            return
        random_input = self._random_input(loop.generator.input_size).to(loop._device)  # pyright: ignore [reportArgumentType]
        result = loop.generator(random_input)
        self.snapshots.append(result)

    def on_epoch_end(self, loop, **kwargs):
        images = self.snapshots[-1]
        show_image(make_grid(images.to('cpu'), nrow=4))

    def _random_input(self, size: int) -> torch.Tensor:
        if size in self._random_inputs:
            return self._random_inputs[size]
        random_input = torch.randn((self.n_images, size, 1, 1))
        self._random_inputs[size] = random_input
        return random_input


class VerboseTrainingCallback:
    """Provide training metrics to standard ouput during training."""

    def __init__(self, batch_interval: int):
        self.batch_interval = batch_interval

    def on_epoch_start(self, loop, epoch: int, **kwargs):
        print(f"Epoch: {epoch}")

    def on_batch_end(self, loop, batch: int, **kwargs):
        if batch % self.batch_interval != 0:
            return
        strings = []
        for key, value in kwargs.items():
            strings.append(f"{key}: {value}")
        print(', '.join(strings))


class HistoryCallback:
    """Create a history of per-epoch results."""

    def __init__(self, batch_end: Iterable[str] = tuple(), epoch_end: Iterable[str] = tuple()):
        self.batch_end = batch_end
        self.epoch_end = epoch_end
        self.history = defaultdict(list)
        self._batch_history = defaultdict(list) # list-defaultdict is created each epoch start.

    def on_epoch_start(self, loop, **kwargs):
        self._batch_history = defaultdict(list)

    def on_batch_end(self, loop, batch: int, *, loss, **kwargs):
        self._batch_history['loss'].append(loss)
        return None

    def on_epoch_end(self, loop, *, test_loss, **kwargs):
        mean_batch_loss = float(np.mean(self._batch_history['loss']))
        self.history['loss'].append(mean_batch_loss)
        self.history['test_loss'].append(test_loss)

    def on_metrics(self, loop, metrics: dict):
        for key, value in metrics.items():
            self.history[key].append(value)

class JSONLoggerCallback:
    """Save training results to a file."""

    def __init__(self, path):
        self.path = path
        try:
            with open(path, 'r') as f:
                logs = json.loads(f.read())
        except FileNotFoundError:
            self._make_logs_dir(path)
            logs = {'train_loss': [], 'test_loss': [], 'test_accuracy': []}
        self.logs = logs
        self._batch_losses = []

    def _make_logs_dir(self, path: str):
        directory, filename = os.path.split(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def on_fit_start(self, loop, train_data: DataLoader, test_data: DataLoader, **kwargs):
        self._data_size = len(test_data.dataset)    # pyright: ignore [reportArgumentType]

    def on_batch_end(self, loop, **kwargs) -> None:
        # collect loss on each batch.
        self._batch_losses.append(kwargs['loss'])

    def on_epoch_end(self, loop, test_loss, **kwargs) -> None:
        # TODO: implement this on metrics.
        # self.logs['test_accuracy'].append(correct / self._data_size)
        self.logs['test_loss'].append(test_loss)
        self._update_training_loss()

    def on_fit_end(self, loop, **kwargs):
        with open(self.path, 'w') as f:
            f.write(json.dumps(self.logs))

    def _update_training_loss(self):
        # append mean loss from batch_losses
        if self._batch_losses:
            mean = np.mean(self._batch_losses)
            self.logs['train_loss'].append(mean)
        self._batch_losses = []


class MetricsCallback:
    """Handle the computation of metrics during training.

    This callback is a publisher as well as a subscriber. At the end of each epoch,
    it publishes to `metrics` on the loop's channel. Callbacks which require metric
    updates should subscribe to the loop's metric messages to receive these updates.
    """
    def __init__(self, metrics: list[Metric]):
        """Initalize a callback that executes metrics."""
        self.metrics = metrics

    def on_batch_end(self, loop, *, y_hat, y_true, **kwargs):
        """Update each metric with the inputs."""
        for metric in self.metrics:
            metric.update(y_hat, y_true)

    def on_epoch_end(self, loop, *, y_hat, y_true, **kwargs):
        """Compute metrics over the training and validation predictions.

        The metrics are published to the loop's channel as `metrics` and
        the payload is a dictionary of the form:
            {'metric': .01, 'metric_val': .02}
        """
        metrics = {}
        for metric in self.metrics:
            name = getattr(metric, 'name', metric.__class__.__name__).lower()
            metrics[name] = metric.compute()
            metric.reset()
            metric.update(y_hat, y_true)
            metrics[name + "_val"] = metric.compute()
            metric.reset()

        loop.publish(loop.name, 'metrics', metrics=metrics)
