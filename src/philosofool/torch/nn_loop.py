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
        self.name = 'gan_loop'
        self.loss = loss
        self._device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # pyright: ignore [reportOptionalMemberAccess]
        self._history = HistoryCallback()
        self._publisher = Publisher()

        self.add_callbacks(self._history)
        self.generator.to(self._device)
        self.discriminator.to(self._device)

        self._publisher.subscribe("gan_loop_control", self)

    @property
    def history(self):
        return self._history.history

    def on_end_epoch(self, publisher, end_fit=True):
        self._end_epoch = True

    def on_end_fit(self, publisher, end_fit=True):
        self._end_fit = True

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
        self.publish(self.name, 'fit_start', data=data)
        self._end_epoch = False
        self._end_fit = False
        for epoch in range(epochs):
            self.publish(self.name, 'epoch_start', epoch=epoch)
            for i, (gen_loss, dis_loss) in enumerate(self.train(data)):
                self.publish(self.name, 'batch_end', batch=i, gen_loss=gen_loss, dis_loss=dis_loss)
                if self._end_epoch:
                    break
            signals = self.publish(self.name, 'epoch_end', epoch=epoch)
            if self._end_fit:
                break
        self.publish(self.name, 'fit_end')

    def add_callbacks(self, *callbacks):
        for callback in callbacks:
            self._publisher.subscribe('gan_loop', callback)

    def publish(self, channel, message, **kwargs) -> list:
        self._publisher.publish(channel, message, self, **kwargs)
        return None


    def train(self, data: DataLoader, train_generator: bool = True, train_discriminator: bool = True) -> Iterator:
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
