from __future__ import annotations

from collections.abc import Iterator

from typing import Any, TYPE_CHECKING, Iterator, Protocol, Callable

import torch
from torch import accelerator, nn

from philosofool.torch.callbacks import HistoryCallback


if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from torch.optim import Optimizer
    Loss = torch.nn.modules.loss._Loss



class TrainingLoop():
    """A loop for handling Pytorch models.

    The loop peforms model optimization and control via callbacks. Internally,
    a loop determines the correct device to run on.

    A training loop:
    - does gradient descent optimization on a model.
    - emits events to a publisher.
    - subscribes callbacks to it's channel.
    - receives events emitted to <loop_name>_control, which can be triggered
      from callbacks.
    """

    def __init__(self, model: nn.Module, optimizer: Optimizer, loss: Loss, name='training_loop'):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.name = name
        self._epochs = 0
        self._device = accelerator.current_accelerator().type if accelerator.is_available() else "cpu"    # pyright: ignore [reportOptionalMemberAccess]
        self._publisher = Publisher()

        self.model.to(self._device)

        self._end_epoch = False
        self._end_fit = False
        self._publisher.subscribe(f'{self.name}_control', self)

    def add_callbacks(self, *callbacks):
        for callback in callbacks:
            self._publisher.subscribe(self.name, callback)

    def subscribe(self, channel, callback):
        """Subscribe callback to channel."""
        self._publisher.subscribe(channel, callback)

    def publish(self, channel, message, **kwargs) -> None:
        """Publish message."""
        self._publisher.publish(channel, message, self, **kwargs)

    def _emit_to_callbacks(self, message, **kwargs) -> None:
        """Publish to self.name.

        This is a private helper to make the steps in fit read a little more clearly.
        """
        self.publish(self.name, message, publisher=self, **kwargs)

    def on_end_epoch(self, publisher, *, end_epoch=True, **kwargs):
        """Listen on subscribed channels for `end_epoch`.

        This event will end the current epoch.
        """
        self._end_epoch = end_epoch

    def on_end_fit(self, publisher, *, end_fit=True, **kwargs):
        """Listen on subscribed channels for `end_fit`.

        This event ends fitting after the current epoch completes; it is expected
        from EarlyStopping and similar callbacks.
        """
        self._end_fit = end_fit

    def process_val_data(self, data: DataLoader) -> tuple[float, torch.Tensor, torch.Tensor]:
        """Return loss, y_hat and y.

        The tensors are detached.
        """
        test_loss = 0.
        y_values, y_hat_values = [], []
        for _, loss, y_hat, y_true in self.process_batches(data, with_grad=False, eval=True):
            test_loss += loss * y_true.shape[0]
            y_values.append(y_true)
            y_hat_values.append(y_hat)

        y_hat = torch.concat(y_hat_values)
        y = torch.concat(y_values)
        n_samples = y_hat.shape[0]
        test_loss = test_loss / n_samples if n_samples else 1.0
        return test_loss, y_hat, y

    def process_batches(self, data: DataLoader, with_grad: bool = True, eval: bool | None = None) -> Iterator:
        """Loop over batches in data and optimize model parameters.

        The loss, predictions and label values are yielded on each batch.
        """

        if eval is None:
            # By default, assume that we want eval mode if and only if we're ignoring gradients.
            eval = not with_grad
        if eval:
            self.model.eval()
        else:
            self.model.train()

        for batch, (X, y) in enumerate(data):
            X, y = X.to(self._device), y.to(self._device)
            loss, pred = self._process_batch(X, y, with_grad)
            yield batch, loss.item(), pred.detach(), y

    def _process_batch(self, X, y, with_grad: bool):
        if with_grad:
            pred = self.model(X)
            loss = self.loss(pred, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            with torch.no_grad():
                X, y = X.to(self._device), y.to(self._device)
                pred = self.model(X)
                loss = self.loss(pred, y)
        return loss, pred



    def fit(self, train_data: DataLoader, test_data: DataLoader, epochs=1, callbacks: list | None = None) -> None:
        """Fit the model to the data.

        Callback cooridinate logging, verbose output, early stopping, etc.

        Events are emitted to a channel of the loop name:
            fit start: train_data, test_data
            epoch_start: epoch
            batch_end: batch, loss, y_hat, y_true
            epoch_end: test_loss, y_hat, y_true (validation metrics)
            fit_end: (None)
        Optionally, `metrics` is emitted by callbacks with a payload of the losses.
        Callbacks should listen on `metrics` to aggregate metrics during the
        loops.
        """
        if callbacks:
            self.add_callbacks(*callbacks)
        self._emit_to_callbacks('fit_start', train_data=train_data, test_data=test_data)
        self._end_epoch = False
        self._end_fit = False
        for epoch in range(self._epochs, self._epochs + epochs):
            self._emit_to_callbacks('epoch_start', epoch=epoch)
            for (batch, loss, y_hat, y) in self.process_batches(train_data):
                # NOTE: Forwarding of loss is unthrilling.
                #       Losses have a dual funciton,
                #       in being metrics (e.g., to control early stopping) and are part
                #       of the gradient descent. We don't want redundant computations.
                #       but the result feels a little convoluted in the separation of responsibilities.
                #       This is the compromise currently.
                self._emit_to_callbacks('batch_end', batch=batch, loss=loss, y_hat=y_hat, y_true=y)
                if self._end_epoch:
                    break
            test_loss, y_hat_val, y_val = self.process_val_data(test_data)
            self._emit_to_callbacks('epoch_end', test_loss=test_loss, y_hat=y_hat_val, y_true=y_val)
            if self._end_fit:
                break
        self._epochs += epochs
        self._emit_to_callbacks('fit_end')

class Publisher:
    """A publisher-subscriber implementation.

    This is used when implementing callbacks in training loops.
    """
    def __init__(self):
        self.topics = {}

    def subscribe(self, topic: str, handler):
        """Add handler to the collection of callables notified when a message is published to topic."""
        if topic not in self.topics:
            self.topics[topic] = set()
        self.topics[topic].add(handler)

    def publish(self, topic, message, *args, **kwargs) -> list:
        """Publish args and kwargs to topic."""
        results = []
        for callback in self.topics.get(topic, []):
            handler = getattr(callback, f'on_{message}', None)
            if handler is None:
                continue
            result = handler(*args, **kwargs)
            if result is not None:
                results.append(result)
        return results
