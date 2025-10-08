from tempfile import TemporaryDirectory
import os
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from philosofool.torch.nn_loop import (
    GANLoop, JSONLoggerCallback, TrainingLoop,
    Publisher, HistoryCallback,
    JSONLoggerCallback,
    EndOnBatchCallback, SnapshotCallback, VerboseTrainingCallback

)
from philosofool.torch.nn_models import Generator, Discriminator

import pytest


def clone_state_dict(module: nn.Module) -> dict:
    """Return a copy of the module state dict, cloning and detaching the tensors in it.

    Useful for checking if a model's state dict has updated.
    """
    return {k: tensor.clone().detach() for k, tensor in module.state_dict().items()}

class CountEpochsCallback:
    """Callback for testing loop counts."""
    def __init__(self):
        self.epochs = 0
        self.batches = 0

    def on_epoch_start(self, loop, epoch, **kwargs):
        self.epochs += 1

    def on_batch_end(self, loop, batch, **kwargs):
        self.batches += 1

class EndAfterOneEpoch:
    def on_epoch_end(self, loop, **kwargs):
        return 'end_fit'

def test_pub_sub(capsys):

    class TestCallback:
        def on_test_message(self, x):
            print(x, end='')

    test_callback = TestCallback()
    publisher = Publisher()
    publisher.subscribe('test', test_callback)
    publisher.publish('test', 'test_message', True)

    assert len(publisher.topics) == 1

    publisher.subscribe('test', test_callback)
    assert len(publisher.topics['test']) == 1, "A handle should only be in a given topic once."

    message = capsys.readouterr().out
    assert message == "True"
    with np.testing.assert_raises(TypeError):
        publisher.publish('test', 'test_message', y=False)


@pytest.fixture
def dataset() -> TensorDataset:
    # 1 batch, 2 rows, three columns
    data = torch.tensor(
        [[1., 0., 0], [0., 1., .0]]
    )
    labels = torch.tensor(
        [[1., 0.], [0., 1.]]
    )
    return TensorDataset(data, labels)


@pytest.fixture
def data_loader(dataset) -> DataLoader:
    data_loader = DataLoader(dataset, batch_size=2)
    return data_loader


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x):
        logits = self.linear(x)
        return logits


@pytest.fixture
def training_loop() -> TrainingLoop:
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=.5)
    loss = nn.CrossEntropyLoss()

    training_loop = TrainingLoop(model, optimizer, loss)
    return training_loop

class TestTrainingLoop:
    def test_fit__callbacks(self, training_loop, data_loader):
        class TestCallback:
            """Track events published, keepinf the order and whether published messages include required information."""
            def __init__(self):
                self.messages = []

            def on_fit_start(self, loop, **kwargs):
                if 'train_data' in kwargs and 'test_data' in kwargs:
                    self.messages.append('fit_start')
                else:
                    self.messages.append('missing train or test data.')

            def on_epoch_start(self, loop, **kwargs):
                if 'epoch' in kwargs:
                    self.messages.append('epoch_start')
                else:
                    self.messages.append('missing epoch in epoch start.')

            def on_batch_end(self, loop, **kwargs):
                if 'batch' in kwargs and 'loss' in kwargs:
                    self.messages.append('batch_end')
                else:
                    self.messages.append("missing keyword argument loss or batch.")

            def on_epoch_end(self, loop, **kwargs):
                if 'correct' in kwargs and 'test_loss' in kwargs:
                    self.messages.append('epoch_end')
                else:
                    self.messages.append('missing kwarg test_loss or correct')

            def on_fit_end(self, loop, **kwargs):
                self.messages.append('fit_end')

        callback = TestCallback()
        training_loop.fit(data_loader, data_loader, epochs=1, callbacks=[callback])
        expected = ['fit_start', 'epoch_start', 'batch_end', 'epoch_end', 'fit_end']
        assert len(callback.messages) <= len(expected), "An event was published more than expected."
        assert len(callback.messages) >= len(expected), "An expected event was not published."
        assert callback.messages == expected, "Some callback received the wrong keywords."


    def test_test(self, training_loop, data_loader):
        correct, loss_value = training_loop.test(data_loader)
        assert training_loop.model.training == False
        assert type(correct) == float
        assert type(loss_value) == float


    def test_fit(self, data_loader, training_loop):

        initial_state = clone_state_dict(training_loop.model)
        train_data, test_data = data_loader, data_loader
        counter = CountEpochsCallback()
        training_loop.fit(train_data, test_data, 8, callbacks=[counter])
        # test if weights updated.
        updated = False
        for original, new in zip(initial_state.items(), training_loop.model.state_dict().items()):
            updated = updated or torch.any(original[1] != new[1])
        assert updated, "Expected some parameters to update."

        assert counter.epochs == 8, "Expected 8 epochs."

    def test_fit__handles_end_fit(self, dataset, training_loop):

        data_loader = DataLoader(dataset, batch_size=1)

        counter = CountEpochsCallback()
        end_on_epoch = EndAfterOneEpoch()
        training_loop.fit(data_loader, data_loader, epochs=3, callbacks=[end_on_epoch, counter])
        assert counter.epochs == 1

    def test_fit__handles_end_epoch(self, dataset, training_loop):
        data_loader = DataLoader(dataset, batch_size=1)

        counter = CountEpochsCallback()
        end_on_batch = EndOnBatchCallback(0)
        training_loop.fit(data_loader, data_loader, epochs=3, callbacks=[end_on_batch, counter])
        assert counter.batches == 3
        assert counter.epochs == 3


    def test_train(self, training_loop, data_loader):

        training_loop.test(data_loader)
        assert training_loop.model.training == False, "Testing the model should set the model training to false."

        loop_iterator = training_loop.train(data_loader)
        last_loss = np.inf
        for batch, loss_value in loop_iterator:
            assert training_loop.model.training == True, "Training the model should set the model training to True."
            assert type(batch) is int
            assert type(loss_value) is float
            assert last_loss > loss_value
            last_loss = loss_value


def test_history_callback():
    publisher = Publisher()
    callback = HistoryCallback()
    publisher.subscribe('event', callback)
    publisher.publish('event', 'epoch_start', None)
    publisher.publish('event', 'batch_end', None, batch=1, val_loss=.1, test_loss=.05)
    publisher.publish('event', 'epoch_end', None)
    assert callback.history == {'val_loss': [.1], 'test_loss': [.05]}

def test_json_logging_callback(data_loader):

    directory = TemporaryDirectory().name
    path = os.path.join(directory, 'logs', 'log.json')
    publisher = Publisher()

    callback = JSONLoggerCallback(path)
    publisher.subscribe('events', callback)
    publisher.publish('events', 'fit_start', None, data_loader, data_loader)
    publisher.publish('events', 'epoch_start', None, epoch=1)
    publisher.publish('events', 'batch_end', None, loss=.1)
    publisher.publish('events', 'epoch_end', None, correct=1, test_loss=.2)
    publisher.publish('events', 'fit_end', None)

    with open(path, 'r') as f:
        logs = json.loads(f.read())
    assert callback.logs == logs
    assert logs['train_loss'] == [.1]
    assert logs['test_loss'] == [.2]
    assert logs['test_accuracy'] == [1 / 2]

@pytest.fixture
def gan_loop() -> GANLoop:
    generator = Generator(5, 2)
    discriminator = Discriminator(2)
    loop = GANLoop(
        generator,
        discriminator,
        torch.optim.SGD(generator.parameters(), .1),
        torch.optim.SGD(discriminator.parameters(), .1),
        nn.BCEWithLogitsLoss()
    )
    return loop

class TestGANLoop:
    def _make_images_fakes(self, loop) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
        """Generate real and fake images for testing, returning tensors and cloned model parameters."""

        images = torch.rand((8, 3, 64, 64)).to(loop._device) / 2 + .5
        generator = loop.generator
        fakes = generator(torch.randn(8, generator.input_size, 1, 1).to(loop._device))
        discriminator = loop.discriminator
        gen_params = {key: tensor.clone().detach() for key, tensor in generator.state_dict().items()}
        dis_params = {key: tensor.clone().detach() for key, tensor in discriminator.state_dict().items()}
        return images, fakes, gen_params, dis_params

    def test_discriminator_step(self, gan_loop):
        images, fakes, gen_params_initial, dis_params_initial = self._make_images_fakes(gan_loop)
        generator = gan_loop.generator
        discriminator = gan_loop.discriminator

        gan_loop.discriminator_step(images, fakes)

        for original_weight, new_weight in zip(gen_params_initial.values(), generator.state_dict().values()):
            assert torch.all(original_weight == new_weight), "Parameters of generator should not update."

        for original_weight, new_weight in zip(dis_params_initial.values(), discriminator.state_dict().values()):
            assert torch.any(original_weight != new_weight.detach()), """Parameters of discriminator should update."""

    def test_generator_step(self, gan_loop):
        images, fakes, gen_params_initial, dis_params_initial = self._make_images_fakes(gan_loop)

        generator = gan_loop.generator
        discriminator = gan_loop.discriminator

        gan_loop.generator_step(fakes, torch.ones(8).to(gan_loop._device))

        updated = False
        for original_weight, new_weight in zip(gen_params_initial.values(), generator.parameters()):
            updated = updated or bool(torch.any(original_weight != new_weight))
        assert updated, "Parameters of generator should update."
        for original_weight, new_weight in zip(dis_params_initial.values(), discriminator.parameters()):
            assert torch.all(original_weight == new_weight.detach()), """Parameters of discriminator should not update."""

    def test_step(self, gan_loop):
        images, fakes, gen_params_initial, dis_params_initial = self._make_images_fakes(gan_loop)

        generator = gan_loop.generator
        discriminator = gan_loop.discriminator

        loader = DataLoader(TensorDataset(images), images.shape[0] // 4)
        n_iterations = 0

        for losses in gan_loop.step(loader):
            n_iterations += 1
        assert n_iterations == 4, "There should be 1 step per batch and there are 4 batches."

        updated = False
        for original, new in zip(gen_params_initial.items(), generator.state_dict().items()):
            key, original_weight = original
            key_new, new_weight = new
            if 'bias' in key:
                continue
            assert torch.any(original_weight != new_weight.detach()), f"Key {key} was not updated."

        for original, new in zip(dis_params_initial.items(), discriminator.state_dict().items()):
            key, original_weight = original
            key_new, new_weight = new

            assert torch.any(original_weight != new_weight.detach()), """Parameters of discriminator should update."""

    def test_fit__callbacks(self, gan_loop):
        class TestCallback:
            def __init__(self):
                self.messages = []

            def on_fit_start(self, loop, **kwargs):
                self.messages.append('fit_start')

            def on_epoch_start(self, loop, **kwargs):
                self.messages.append('epoch_start')

            def on_batch_end(self, loop, **kwargs):
                self.messages.append('batch_end')

            def on_epoch_end(self, loop, **kwargs):
                self.messages.append('epoch_end')

            def on_fit_end(self, loop, **kwargs):
                self.messages.append('fit_end')

        images = torch.rand((64, 3, 64, 64)) / 2 + .5
        loader = DataLoader(TensorDataset(images), 32)
        test_callback = TestCallback()
        gan_loop.fit(loader, epochs=1, callbacks=[test_callback])
        expected = ['fit_start', 'epoch_start', 'batch_end', 'batch_end', 'epoch_end', 'fit_end']
        assert test_callback.messages == expected, f"Callback expected {expected}, but received {test_callback.messages}"

    def test_fit__handles_end_epoch(self, gan_loop: GANLoop):
        images, fakes, _, __ = self._make_images_fakes(gan_loop)
        counter = CountEpochsCallback()
        end_on_batch = EndOnBatchCallback(1)
        dataset = TensorDataset(images)
        data_loader = DataLoader(dataset, 2)
        gan_loop.fit(data_loader, epochs=2, callbacks=[end_on_batch, counter])
        assert counter.batches == 4, "We expect two batches per epoch."
        assert counter.epochs == 2, "We expect two epochs."

    def test_fit__handles_end_fit(self, gan_loop: GANLoop):
        images, fakes, _, __ = self._make_images_fakes(gan_loop)
        counter = CountEpochsCallback()
        end_after_epoch = EndAfterOneEpoch()
        dataset = TensorDataset(images)
        data_loader = DataLoader(dataset, 2)
        gan_loop.fit(data_loader, epochs=2, callbacks=[end_after_epoch, counter])
        assert counter.batches == 4
        assert counter.epochs == 1



def test_end_on_batch():
    end_on_batch = EndOnBatchCallback(2)
    assert end_on_batch.on_batch_end(None, batch=2, gen_loss=.1) == 'end_epoch', "Should emit 'end_epoch' signal, gen_loss (unused) is accepeccted."
    assert end_on_batch.on_batch_end(None, batch=1) is None, "Should not end on batch one. Missing gen_loss is accepted."

def test_snapshot_callback():
    generator = Generator(10, 10)
    discriminator = Discriminator(10)

    loss = nn.BCEWithLogitsLoss()
    loop = GANLoop(
        generator,
        discriminator,
        torch.optim.SGD(generator.parameters(), lr=.01, momentum=0),
        torch.optim.SGD(discriminator.parameters(), lr=.01, momentum=0),
        loss
    )

    loop.generator.eval()
    snapshot_callback = SnapshotCallback(n_images=1, interval=2)
    snapshot_callback.on_batch_end(loop, batch=2, gen_loss=.1, dis_loss=.1)
    assert len(snapshot_callback.snapshots ) == 1, "Snapshots should update when called on batch interval."
    snapshot_callback.on_batch_end(loop, batch=3, gen_loss=.1, dis_loss=.1)
    assert len(snapshot_callback.snapshots ) == 1, "Snapshots should update only on interval."
    snapshot_callback.on_batch_end(loop, batch=2, gen_loss=.1, dis_loss=.1)
    assert len(snapshot_callback.snapshots) == 2, "Snapshots should update when called on batch interval."
    snapshots = snapshot_callback.snapshots
    assert torch.allclose(snapshots[0].detach(), snapshots[1].detach())

    snapshot_callback.on_epoch_end(loop, epoch=1)
    assert snapshot_callback.snapshots[0].shape[0] == snapshot_callback.n_images == 1

def test_vebose_training_callback():

    callback = VerboseTrainingCallback(batch_interval=2)
    callback.on_batch_end(None, batch=1)
    callback.on_batch_end(None, batch=2, gen_loss=.1, dis_loss=.2)
    callback.on_epoch_start(None, epoch=12, unused_argument='ignored')