from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
import json
from typing import Any, TYPE_CHECKING, Iterator, Protocol
import warnings

import numpy as np
import torch
from torch import accelerator
from torch import nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from torch.optim import Optimizer
    Loss = torch.nn.modules.loss._Loss


def summarize_training(batch: int, loss: float, data: DataLoader) -> str:
    batch_size = data.batch_size
    size = len(data.dataset)    # pyright: ignore [reportArgumentType]
    loss, current = loss, (batch + 1) * batch_size    # pyright: ignore [reportOperatorIssue]
    sample_length = int(np.ceil(np.log10(size)))
    return f"loss: {loss:>7f}  [{current:>{sample_length}d}/{size:>{sample_length}d}]"

def summarize_test(correct: float, loss: float) -> str:
    return f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {loss:>8f} \n"


class TrainingLogger(Protocol):
    def training(self, batch: int, loss: float, *metrics: float) -> None:
        ...

    def testing(self, correct: float, loss: float) -> None:
        ...

    def start_epoch(self, epoch: int) -> None:
        ...

    def finish(self):
        ...

class StandardOutputLogger:
    """Handle logging to standard output."""
    def __init__(self, interval: int = 1):
        self.interval = interval
        self._losses = []

    def training(self, batch: int, loss: float, data: DataLoader) -> None:
        """Log the outputs of a training loop to standard output."""
        self._losses.append(loss)
        if batch % self.interval == 0:
            print(summarize_training(batch, np.mean(self._losses), data))  # pyright: ignore [reportArgumentType]

    def testing(self, correct: float, loss: float) -> None:
        """Log the outputs of testing to standard output."""
        print(summarize_test(correct, loss))

    def start_epoch(self, epoch: int) -> None:
        self._losses = []
        print(f"Epoch: {epoch}")

    def finish(self):
        return

class JSONLogger:
    """Save training results to a file."""
    def __init__(self, path):
        self.path = path
        try:
            with open(path, 'r') as f:
                logs = json.loads(f.read())
            self.logs = logs
        except FileNotFoundError:
            self.logs = {'train_loss': [], 'test_loss': [], 'test_accuracy': []}
        self._epoch_training = []

    def start_epoch(self, epoch):
        self._update_epochs()

    def training(self, batch, loss, data) -> None:
        self._epoch_training.append(loss)

    def testing(self, correct, loss) -> None:
        self.logs['test_accuracy'].append(correct)
        self.logs['test_loss'].append(loss)

    def finish(self):
        self._update_epochs()
        with open(self.path, 'w') as f:
            f.write(json.dumps(self.logs))

    def _update_epochs(self):
        if self._epoch_training:
            mean = np.mean(self._epoch_training)
            self.logs['train_loss'].append(mean)
        self._epoch_training = []

class CompositeLogger:
    """Compose simple loggers."""
    def __init__(self, *loggers: TrainingLogger):
        self.loggers = loggers

    def training(self, batch, loss, size):
        for logger in self.loggers:
            logger.training(batch, loss, size)

    def testing(self, correct, loss):
        for logger in self.loggers:
            logger.testing(correct, loss)

    def start_epoch(self, epoch):
        for logger in self.loggers:
            logger.start_epoch(epoch)

    def finish(self):
        for logger in self.loggers:
            logger.finish()

class TrainingLoop():
    def __init__(self, model: nn.Module, optimizer: Optimizer, loss: Loss, logging: TrainingLogger | None = None):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.logging = logging or StandardOutputLogger()
        self._epochs = 0
        self._device = accelerator.current_accelerator().type if accelerator.is_available() else "cpu"    # pyright: ignore [reportOptionalMemberAccess]
        self.model.to(self._device)

    def test(self, data: DataLoader) -> tuple:
        size = len(data.dataset)  # pyright: ignore [reportArgumentType]
        num_batches = len(data)
        self.model.eval()
        test_loss, correct = 0., 0.
        with torch.no_grad():
            for X, y in data:
                X, y = X.to(self._device), y.to(self._device)
                pred = self.model(X)
                test_loss += self.loss(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        return correct, test_loss

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
            yield batch, loss.item()

    def fit(self, train_data: DataLoader, test_data: DataLoader, epochs=1) -> None:
        for epoch in range(self._epochs, self._epochs + epochs):
            self.logging.start_epoch(epoch)
            for (batch, loss) in self.train(train_data):
                self.logging.training(batch, loss, train_data)
            test_correct, test_loss = self.test(test_data)
            self.logging.testing(test_correct, test_loss)
        self._epochs += epochs
        self.logging.finish()


class Callback(Protocol):
    on: str

    def __call__(self, loop, **kwargs):
        ...

class LambdaCallback:
    def __init__(self, on: str, fn: Callable):
        self.on = on
        self._fn = fn

    def __call__(self, loop, **kwargs):
        return self._fn(loop, **kwargs)

class HistoryCallback:
    def __init__(self):
        self.on = 'batch_end'
        self.history = defaultdict(list)

    def __call__(self, loop, **kwargs):
        for key, value in kwargs.items():
            self.history[key].append(value)
        return None

class Publisher:
    """A publisher-subscriber implementation.

    This is used when implementing callbacks in training loops.
    """
    def __init__(self):
        self.topics = {}

    def subscribe(self, topic: str, handler: Callable):
        """Add handler to the collection of callables notified when a message is published to topic."""
        if topic not in self.topics:
            self.topics[topic] = set()
        self.topics[topic].add(handler)

    def publish(self, topic, *args, **kwargs) -> list:
        """Publish args and kwargs to topic."""
        results = []
        if topic not in self.topics:
            return []
        for handler in self.topics[topic]:
            result = handler(*args, **kwargs)
            if result is not None:
                results.append(result)
        return results

class GANLoop:
    """A training loop for Generative Adversarial Networks.

    This class manages the training of both discriminator and generator.
    """
    def __init__(
            self,
            generator: nn.Module,
            discriminator: nn.Module,
            generator_optimizer: Optimizer,
            discriminator_optimizer: Optimizer,
            loss: Loss
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optim = generator_optimizer
        self.discriminator_optim = discriminator_optimizer
        self.loss = loss
        self._device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self._history = HistoryCallback()
        self._publisher = Publisher()
        self.generator.to(self._device)
        self.discriminator.to(self._device)

    def fit(self, data: DataLoader, epochs: int = 1, callbacks: list[Callback] = []):
        gen_loss, dis_loss = None, None
        self._subscribe_callbacks(callbacks)

        for epoch in range(epochs):
            self._publish('epoch_start', epoch=epoch)
            for i, (gen_loss, dis_loss) in enumerate(self.step(data)):
                signals = self._publish('batch_end', batch=i, gen_loss=gen_loss, dis_loss=dis_loss)
                if 'end_batch' in signals:
                    break
            signals = self._publish('epoch_end', epoch=epoch)
            if 'end_fit' in signals:
                break
        self._publish('fit_end')

    def _subscribe_callbacks(self, callbacks):
        for callback in [self._history] + callbacks:
            self._publisher.subscribe(callback.on, callback)

    def _publish(self, message, **kwargs) -> list:
        if self._publisher is None:
            return []
        signals = self._publisher.publish(message, self, **kwargs)
        return signals


    def step(self, data: DataLoader, train_generator: bool = True, train_discriminator: bool = True) -> Iterator:
        """Perform one step of generator and discriminator optimization, yielding loss on each batch."""
        self.generator.to(self._device)
        self.discriminator.to(self._device)
        self.generator.train()
        self.discriminator.train()

        # assume loss == 1 when no training is requested.
        generator_loss = torch.tensor(1.)
        discriminator_loss = torch.tensor(1.)

        for images in data:
            if isinstance(images, list):
                images = images[0]
                assert isinstance(images, torch.Tensor), "Images is expected to be a tensor."
            batch_size = images.shape[0]

            # use "real" label when generating genertor loss
            gen_labels = torch.ones(batch_size).to(self._device)

            random_input = torch.randn(batch_size, self.generator.input_size, 1, 1).to(self._device)   # pyright: ignore [reportArgumentType]
            images = images.to(self._device)
            fakes = self.generator(random_input).to(self._device)

            if train_discriminator:
                discriminator_loss = self.discriminator_step(images, fakes)

            if train_generator:
                generator_loss = self.generator_step(fakes, gen_labels)

            yield generator_loss.item(), discriminator_loss.item()

    def discriminator_step(self, images, fakes):
        batch_size = images.shape[0]
        self.discriminator.train()
        self.generator.eval()
        self.discriminator_optim.zero_grad()

        images_pred = self.discriminator(images).to(self._device).view(-1)
        loss_real = self.loss(images_pred, torch.ones(batch_size).to(self._device))
        fake_pred = self.discriminator(fakes.detach()).to(self._device).view(-1)
        loss_fake = self.loss(fake_pred, torch.zeros(batch_size).to(self._device))

        loss_fake.backward()
        loss_real.backward()
        loss = loss_real + loss_fake
        self.discriminator_optim.step()

        return loss
        # print(self.discriminator.state_dict()['residual_layer.0.weight'][0, 0, 0])

    def generator_step(self, fakes, gen_labels):
        self.discriminator.eval()
        self.generator.train()
        self.generator_optim.zero_grad()
        discriminator_pred = self.discriminator(fakes)
        generator_loss = self.loss(discriminator_pred.view(-1), gen_labels)
        generator_loss.backward()
        self.generator_optim.step()
        return generator_loss

    def save_checkpoint(self, path: str, meta: dict | None):
        checkpoint = {
            'generator': self.generator,
            'discriminator': self.discriminator,
            'generator_optimizer': self.generator_optim,
            'discriminator_optimizer': self.discriminator_optim,
            'loss': self.loss
        }
        if meta:
            checkpoint['meta'] = meta

        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path) -> tuple['GANLoop', dict | None]:
        checkpoint = torch.load(path, weights_only=False)
        meta = checkpoint.pop('meta', None)
        # NOTE: handling a typo in a previous version.
        if 'genertator_optimizer' in checkpoint:
            checkpoint['generator_optimizer'] = checkpoint.pop('genertator_optimizer')
        return cls(**checkpoint), meta
