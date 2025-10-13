from __future__ import annotations
from collections import defaultdict
import json
import os
import numpy as np
from typing import TYPE_CHECKING



import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from philosofool.torch.visualize import show_image

if TYPE_CHECKING:
    from philosofool.torch.nn_loop import GANLoop



class EndOnBatchCallback:
    """After a number of batches, proceed to the next epoch."""
    def __init__(self, last_batch: int):
        self.last_batch = last_batch

    def on_batch_end(self, loop, batch: int, **kwargs):
        if batch == self.last_batch:
            return 'end_epoch'


class SnapshotCallback:
    """Collect snapshots on an interval."""
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
    def __init__(self):
        self.history = defaultdict(list)
        self._batch_history = defaultdict(list) # list-defaultdict is created each epoch start.

    def on_epoch_start(self, loop, **kwargs):
        self._batch_history = defaultdict(list)

    def on_batch_end(self, loop, batch: int, **kwargs):
        for key, value in kwargs.items():
            self._batch_history[key].append(value)
        return None

    def on_epoch_end(self, loop, **kwargs):
        for key, values in self._batch_history.items():
            self.history[key].append(float(np.mean(values)))
        for key, value in kwargs.items():
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

    def on_epoch_end(self, loop, correct, test_loss, **kwargs) -> None:
        self.logs['test_accuracy'].append(correct / self._data_size)
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