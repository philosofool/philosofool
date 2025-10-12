from __future__ import annotations

from collections import defaultdict
from typing import Any, TYPE_CHECKING, Iterator, Protocol, Callable
import json
import os

import numpy as np
import torch
from torch import accelerator, nn
from torchvision.utils import make_grid

from philosofool.torch.visualize import show_image

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


class TrainingLoop():
    def __init__(self, model: nn.Module, optimizer: Optimizer, loss: Loss):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self._epochs = 0
        self._device = accelerator.current_accelerator().type if accelerator.is_available() else "cpu"    # pyright: ignore [reportOptionalMemberAccess]
        self._history = HistoryCallback()
        self._publisher = Publisher()

        self.add_callbacks(self._history)

        self.model.to(self._device)

    @property
    def history(self) -> dict:
        return self._history.history

    def add_callbacks(self, *callbacks):
        for callback in callbacks:
            self._publisher.subscribe('training_loop', callback)

    def _publish(self, message, **kwargs) -> list:
        if self._publisher is None:
            return []
        signals = self._publisher.publish('training_loop', message, self, **kwargs)
        return signals

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

    def fit(self, train_data: DataLoader, test_data: DataLoader, epochs=1, callbacks: list | None = None) -> None:
        if callbacks:
            self.add_callbacks(*callbacks)
        self._publish('fit_start', train_data=train_data, test_data=test_data)
        for epoch in range(self._epochs, self._epochs + epochs):
            self._publish('epoch_start', epoch=epoch)
            for (batch, loss) in self.train(train_data):
                signals = self._publish('batch_end', batch=batch, loss=loss)
                if 'end_epoch' in signals:
                    break
            test_correct, test_loss = self.test(test_data)
            signals = self._publish('epoch_end', correct=test_correct, test_loss=test_loss)
            if 'end_fit' in signals:
                break
        self._epochs += epochs
        self._publish('fit_end')

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
        self._device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # pyright: ignore [reportOptionalMemberAccess]
        self._history = HistoryCallback()
        self._publisher = Publisher()

        self.add_callbacks(self._history)
        self.generator.to(self._device)
        self.discriminator.to(self._device)

    @property
    def history(self):
        return self._history.history

    def fit(self, data: DataLoader, epochs: int = 1, callbacks: list = []):
        """
        Train the GAN for a given number of epochs.

        This method coordinates the GAN training loop, publishing events to any
        registered callbacks at key points (epoch start, batch end, epoch end, fit end).
        Callbacks can be used for logging, adaptive learning rate scheduling,
        early stopping, or other custom behavior.

        Parameters
        ----------
        data : torch.utils.data.DataLoader
            The dataloader providing real training samples for the GAN.
        epochs : int, default=1
            Number of epochs to train for.
        callbacks : list[Callback], optional
            A list of callback objects that subscribe to training events. Each callback
            should implement event handlers compatible with the topics published by
            the training loop (e.g. "epoch_start", "batch_end", "epoch_end", "fit_end").

        Events Published
        ----------------
        - "fit_start": at the start of training, with no arguments.
        - "epoch_start": at the start of each epoch, with arguments {epoch}.
        - "batch_end": after each batch, with arguments {batch, gen_loss, dis_loss}.
        If any callback returns the signal "end_batch", the batch loop is interrupted.
        - "epoch_end": after each epoch, with arguments {eptestoch}.
        If any callback returns the signal "end_fit", training terminates early.
        - "fit_end": after the full training loop has finished (normally or early).

        Notes
        -----
        - Generator and discriminator losses are yielded from `self.step(data)`,
        which should define the per-batch training behavior.
        - The last values of `gen_loss` and `dis_loss` from the final batch are
        retained in local scope but not returned.
        - Callbacks provide the only mechanism for logging, checkpointing, or
        early termination; the loop itself only publishes events.

        Examples
        --------
        >>> trainer.fit(train_loader, epochs=50, callbacks=[LoggingCallback(), EarlyStopping()])
        """
        gen_loss, dis_loss = None, None
        self.add_callbacks(*callbacks)
        self._publish('fit_start', data=data)
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

    def add_callbacks(self, *callbacks):
        for callback in callbacks:
            self._publisher.subscribe('gan_loop', callback)

    def _publish(self, message, **kwargs) -> list:
        if self._publisher is None:
            return []
        signals = self._publisher.publish('gan_loop', message, self, **kwargs)
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
        return cls(**checkpoint), meta


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
