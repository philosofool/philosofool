from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterable
import json
import os
import numpy as np
from typing import TYPE_CHECKING
import warnings

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from philosofool.torch.visualize import show_image
from philosofool.torch.metrics import Metric

if TYPE_CHECKING:
    from philosofool.torch.nn_loop import TrainingLoop
    from philosofool.torch.experimental.nn_loop import GANLoop


class EarlyStoppingCallabck:
    """Handle early stopping of training loops."""

    def __init__(self, patience: int, monitor: str = 'test_loss', refresh_on_new_fit: bool = True):
        if monitor == 'test_loss':
            warnings.warn("The default value for monitor will switch to 'loss_val' in a future version.")
        self.patience = patience
        if monitor not in {'loss_val', 'test_loss'}:
            raise NotImplementedError("EarlyStopping currently only suppots stopping on validation loss.")
        self.monitor = monitor
        self._loops_since_improvement = 0
        self._best = None
        self.refresh_on_new_fit = refresh_on_new_fit

    def on_fit_start(self, loop, **kwargs):
        """Refresh so it takes patiences iterations before stopping the loop."""
        if self.refresh_on_new_fit:
            self._loops_since_improvement = 0
            self._best = None

    def on_batch_end(self, loop, batch, loss, **kwargs):
        ...

    def on_epoch_end(self, loop, **kwargs):
        """Emit 'end_fit' if there has been no improvement."""
        test_loss = kwargs.get(self.monitor)

        if test_loss is None:
            raise KeyError("Monitored metric not in arguments.")
        self._determine_improvement(loop, test_loss)
        if self._loops_since_improvement >= self.patience:
            loop.publish(f"{loop.name}_control", 'end_fit')

        # TODO: implement early stoping on metrics events.

    def _determine_improvement(self, loop, target_metric):
        if self._best is None:
            self._best = target_metric
            return
        if target_metric < self._best:
            self._best = target_metric
            self._loops_since_improvement = 0
            return
        self._loops_since_improvement += 1


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

    # TODO: configure this with MetricsCallback to work with on_metrics.
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

    def __init__(self):
        self.history = defaultdict(list)
        self._batch_history = defaultdict(list) # list-defaultdict is created each epoch start.

    def on_epoch_start(self, loop, **kwargs):
        self._batch_history = defaultdict(list)

    def on_batch_end(self, loop, batch: int, **kwargs):
        for key, value in kwargs.items():
            if 'loss' not in key:
                continue
            if key != 'loss':
                warnings.warn("Passing losses by names other than 'loss', e.g, by 'train_loss' is deprecated.")
            self._batch_history[key].append(value)
        return None

    def on_epoch_end(self, loop, **kwargs):
        """Update history by aggregating the batch results other losses."""
        for key, value in self._batch_history.items():
            if 'loss' not in key:
                continue
            value_mean = float(np.mean(value))
            self.history[key].append(value_mean)
        for key, value in kwargs.items():
            if 'loss' not in key:
                continue
            self.history[key].append(value)
        pass

    def on_metrics(self, loop, metrics: dict):
        for key, value in metrics.items():
            self.history[key].append(value)

class JSONLoggerCallback:
    """Save training results to a file."""

    def __init__(self, path):
        warning_msg = """{self.__class__.__name__} is scheduled for revisions in a future version.
        All metrics names will use a <metric_name> and <metric_name>_val format.
        Use caution when linking processes to the JSON files this logger writes.
        """
        warnings.warn(warning_msg, category=FutureWarning, stacklevel=2)
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
        """Initalize a callback that executes metrics.

        This callback always captures training loss and, if processed,
        validation loss, even if the rest of the metrics are empty.
        """
        self.metrics = metrics
        self._batch_losses = []
        self._n_samples = .0

    def on_batch_end(self, loop, *, y_hat, y_true, **kwargs):
        """Update each metric with the inputs."""
        for metric in self.metrics:
            metric.update(y_hat, y_true)
        if 'train_loss' in kwargs:
            warnings.warn(
                """Passing loss by name 'train_loss' is deprecated and will be removed in a future version.
                Use bare 'loss' when passing loss metrics. Batch losses should be training losses.""",
                category=PendingDeprecationWarning,
                stacklevel=2)
            self._batch_losses.append(kwargs['test_loss'] * y_hat.shape[0])
        else:
            self._batch_losses.append(kwargs['loss'] * y_hat.shape[0])
        self._n_samples += y_hat.shape[0]


    def on_epoch_end(self, loop, *, y_hat, y_true, **kwargs):
        """Compute metrics over the training and validation predictions.

        The metrics are published to the loop's channel as `metrics` and
        the payload is a dictionary of the form:
            {'metric': .01, 'metric_val': .02}
        """
        metrics = {}
        if 'test_loss' in kwargs:
            warnings.warn(
                """Passing loss by name 'test_loss' is deprecated and will be removed in a future version.
                Use bare 'loss' when passing loss metrics. End epoch metrics should be valiation losses.""",
                category=PendingDeprecationWarning,
                stacklevel=2)
            metrics['loss_val'] = kwargs['test_loss']
        elif 'loss' in kwargs:
            metrics['loss_val'] = kwargs['loss']
        metrics['loss'] = np.sum(self._batch_losses) / self._n_samples
        self._batch_losses = []
        self._n_samples = 0.
        for metric in self.metrics:
            name = getattr(metric, 'name', metric.__class__.__name__).lower()
            metrics[name] = metric.compute()
            metric.reset()
            metric.update(y_hat, y_true)
            metrics[name + "_val"] = metric.compute()
            metric.reset()

        loop.publish(loop.name, 'metrics', metrics=metrics)
