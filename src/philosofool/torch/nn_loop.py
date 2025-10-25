from __future__ import annotations

from typing import Any, TYPE_CHECKING, Iterator, Protocol, Callable

import torch
from torch import accelerator, nn

from philosofool.torch.callbacks import HistoryCallback


if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from torch.optim import Optimizer
    Loss = torch.nn.modules.loss._Loss



class TrainingLoop():
    def __init__(self, model: nn.Module, optimizer: Optimizer, loss: Loss, name='training_loop'):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.name = name
        self._epochs = 0
        self._device = accelerator.current_accelerator().type if accelerator.is_available() else "cpu"    # pyright: ignore [reportOptionalMemberAccess]
        self._history = HistoryCallback()
        self._publisher = Publisher()

        self.add_callbacks(self._history)

        self.model.to(self._device)

        self._end_epoch = False
        self._end_fit = False
        self._publisher.subscribe(f'{self.name}_control', self)

    @property
    def history(self) -> dict:
        return self._history.history

    def add_callbacks(self, *callbacks):
        for callback in callbacks:
            self._publisher.subscribe(self.name, callback)

    def subscribe(self, channel, callback):
        self._publisher.subscribe(channel, callback)

    def publish(self, channel, message, **kwargs) -> list:
        self._publisher.publish(channel, message, self, **kwargs)

    def _emit_to_callbacks(self, message, **kwargs) -> list:
        """Publish to self.name.

        This is a private helper to make the steps in fit read a little more clearly.
        """
        self.publish(self.name, message, publisher=self, **kwargs)

    def on_end_epoch(self, publisher, *, end_epoch=True, **kwargs):
        self._end_epoch = end_epoch

    def on_end_fit(self, publisher, *, end_fit=True, **kwargs):
        self._end_fit = end_fit

    def test(self, data: DataLoader) -> tuple[float, torch.Tensor, torch.Tensor]:
        """Return loss, y_hat and y.

        The tensors are detached.
        """
        size = len(data.dataset)  # pyright: ignore [reportArgumentType]
        num_batches = len(data)
        self.model.eval()
        test_loss = 0.
        y_values, y_hat_values = [], []
        n_samples = 0
        with torch.no_grad():
            for X, y in data:
                X, y = X.to(self._device), y.to(self._device)
                y_hat = self.model(X)
                loss = self.loss(y_hat, y).item()
                n_samples += y.shape[0]
                test_loss += loss * y.shape[0]
                y_values.append(y)
                y_hat_values.append(y_hat)
        test_loss = test_loss / n_samples
        y_hat = torch.concat(y_hat_values)
        y = torch.concat(y_values)
        return test_loss, y_hat, y

    def train(self, data: DataLoader) -> Iterator:
        size = len(data.dataset)  # pyright: ignore [reportArgumentType]
        self.model.train()
        for batch, (X, y) in enumerate(data):
            X, y = X.to(self._device), y.to(self._device)
            pred = self.model(X)
            loss = self.loss(pred, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            yield batch, loss.item(), pred.detach(), y

    def fit(self, train_data: DataLoader, test_data: DataLoader, epochs=1, callbacks: list | None = None) -> None:
        if callbacks:
            self.add_callbacks(*callbacks)
        self._emit_to_callbacks('fit_start', train_data=train_data, test_data=test_data)
        self._end_epoch = False
        self._end_fit = False
        for epoch in range(self._epochs, self._epochs + epochs):
            self._emit_to_callbacks('epoch_start', epoch=epoch)
            for (batch, loss, y_hat, y) in self.train(train_data):
                # NOTE: Forwarding of loss is unthrilling.
                #       Losses have a dual funciton,
                #       in being metrics (e.g., to control early stopping) and are part
                #       of the gradient descent. We don't want redundant computations.
                #       but the result feels a little convoluted in the separation of responsibilities.
                #       This is the compromise currently.
                self._emit_to_callbacks('batch_end', batch=batch, loss=loss, y_hat=y_hat, y_true=y)
                if self._end_epoch:
                    break
            test_loss, y_hat_val, y_val = self.test(test_data)
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

