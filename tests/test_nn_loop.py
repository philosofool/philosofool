from sympy import Ge
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from philosofool.torch.nn_loop import (
    GANLoop, JSONLogger, StandardOutputLogger, TrainingLoop,
    Publisher, HistoryCallback,
    JSONLogger, StandardOutputLogger, CompositeLogger,
    EndOnBatchCallback, SnapshotCallback, VerboseTrainingCallback

)
from philosofool.torch.nn_models import Generator, Discriminator
import numpy as np

import pytest

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
def data_loader() -> DataLoader:
    # 1 batch, 2 rows, three columns
    data = torch.tensor(
        [[1., 0., 0], [0., 1., .0]]
    )
    labels = torch.tensor(
        [[1., 0.], [0., 1.]]
    )
    data_loader = DataLoader(TensorDataset(data, labels), batch_size=2)
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


def test_train_classifier__test(training_loop, data_loader):
    correct, loss_value = training_loop.test(data_loader)
    assert training_loop.model.training == False
    assert type(correct) == float
    assert type(loss_value) == float


def test_train_classifier__fit_logging(capsys, data_loader):
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=.5)
    loss = nn.CrossEntropyLoss()

    training_loop = TrainingLoop(model, optimizer, loss, logging=None)
    assert isinstance(training_loop.logging, StandardOutputLogger)
    training_loop.fit(data_loader, data_loader)
    assert 'loss' in capsys.readouterr().out


def test_json_logger():
    from tempfile import TemporaryDirectory
    import os
    file = TemporaryDirectory()
    path = os.path.join(file.name, 'test.json')
    logger = JSONLogger(path)

    logger.start_epoch(1)
    logger.testing(.57, .45)
    logger.training(64, .50, 1)
    logger.finish()

    logs = logger.logs
    assert logs['train_loss'] == [.5]
    assert logs['test_loss'] == [.45]
    assert logs['test_accuracy'] == [.57]

    logs = JSONLogger(path).logs
    assert logs['train_loss'] == [.5]
    assert logs['test_loss'] == [.45]
    assert logs['test_accuracy'] == [.57]

def test_train_classifier__fit(data_loader, training_loop):
    train_data, test_data = data_loader, data_loader
    training_loop.fit(train_data, test_data, 8)

def test_standard_output_logger(capsys):
    logger = StandardOutputLogger()
    data = DataLoader(TensorDataset(
            torch.rand((256, 1)),
            torch.rand((256, 1))),
        batch_size=64)
    logger.training(2, .0001, data)
    captured = capsys.readouterr().out
    assert "loss: 0.0001" in captured, "The loss should be printed to std out."
    assert f"[{3 * 64}/256" in captured

    logger.training(0, .0003, data)
    captured = capsys.readouterr().out
    assert "loss: 0.0002" in captured, "New loss should be the average of observed losses."
    assert f"[ 64/256" in captured

    logger.start_epoch(42)
    assert 'Epoch: 42' in capsys.readouterr().out

def test_composed_logger(capsys):
    from tempfile import TemporaryDirectory
    import os
    data = DataLoader(
        TensorDataset(
            torch.rand((256, 1)),
            torch.rand((256, 1))),
        batch_size=64)

    tempdir = TemporaryDirectory()
    directory = tempdir.name
    file1 = os.path.join(directory, 'test1.json')
    file_logger = JSONLogger(file1)
    logger = CompositeLogger(file_logger, StandardOutputLogger())

    logger.training(2, .0001, data)
    logger.finish()
    assert 'loss' in capsys.readouterr().out, "Standard logger should log to the output."
    assert file_logger.logs['train_loss'] == [.0001], "File logger should also log the loss."

    logger.testing(.95, .001)
    assert 'Accuracy' in capsys.readouterr().out, "Standard logger should log Accuracy to the standard output."
    assert file_logger.logs['test_accuracy'] == [.95], "The file logger should capture the test accuracy."

    logger.start_epoch(42)
    assert 'Epoch: 42' in capsys.readouterr().out


def test_train_classifier__train(training_loop, data_loader):

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
    publisher.publish('event', 'batch_end', None, batch=1, val_loss=.1, test_loss=.05)
    assert callback.history == {'val_loss': [.1], 'test_loss': [.05]}

@pytest.fixture
def gan_loop() -> GANLoop:
    generator = Generator(5, 2)
    discriminator = Discriminator(2)
    loop = GANLoop(
        generator,
        discriminator,
        torch.optim.SGD(generator.parameters(), .01),
        torch.optim.SGD(discriminator.parameters(), .01),
        nn.BCEWithLogitsLoss()
    )
    return loop

class TestGANLoop:
    def test_discriminator_step(self, gan_loop):
        images = torch.rand((8, 3, 64, 64)) / 2 + .5
        fakes = torch.rand((8, 3, 64, 64)) / 2
        generator = gan_loop.generator
        discriminator = gan_loop.discriminator
        gen_params_initial = [tensor.clone().detach() for tensor in generator.parameters()]
        dis_params_initial = [tensor.clone().detach() for tensor in discriminator.parameters()]

        gan_loop.discriminator_step(images, fakes)

        for original_weight, new_weight in zip(gen_params_initial, generator.parameters()):
            assert torch.all(original_weight == new_weight), "Parameters of generator should not update."

        for original_weight, new_weight in zip(dis_params_initial, discriminator.parameters()):
            assert torch.any(original_weight != new_weight.detach()), """Parameters of discriminator should update."""

    def test_generator_step(self, gan_loop):
        images = torch.rand((8, 3, 64, 64)).to(gan_loop._device) / 2 + .5
        fakes = torch.rand((8, 3, 64, 64)).to(gan_loop._device) / 2
        generator = gan_loop.generator
        discriminator = gan_loop.discriminator
        gen_params_initial = [tensor.clone().detach() for tensor in generator.parameters()]
        dis_params_initial = [tensor.clone().detach() for tensor in discriminator.parameters()]

        gen_params_initial = [tensor.clone().detach() for tensor in generator.parameters()]
        dis_params_initial = [tensor.clone().detach() for tensor in discriminator.parameters()]
        gan_loop.generator_step(fakes, torch.ones(8).to(gan_loop._device))

        updated = False
        for original_weight, new_weight in zip(gen_params_initial, generator.parameters()):
            updated = updated or bool(torch.any(original_weight.detach() != new_weight.detach()))
        assert updated, "Parameters of generator should update."
        for original_weight, new_weight in zip(dis_params_initial, discriminator.parameters()):
            assert torch.all(original_weight == new_weight.detach()), """Parameters of discriminator should not update."""

    def test_step(self, gan_loop):
        from torch.utils.data import TensorDataset, DataLoader

        images = torch.rand((64, 3, 64, 64)) / 2 + .5
        fakes = torch.rand((64, 3, 64, 64)) / 2 + .01
        generator = gan_loop.generator
        discriminator = gan_loop.discriminator
        gen_params_initial = {k: tensor.clone().detach() for k, tensor in generator.state_dict().items()}
        dis_params_initial = {k: tensor.clone().detach() for k, tensor in discriminator.state_dict().items()}
        loader = DataLoader(TensorDataset(images), 16)
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


def test_end_on_batch():
    end_on_batch = EndOnBatchCallback(2)
    assert end_on_batch.on_batch_end(None, batch=2, gen_loss=.1) == 'end_batch', "Should emit 'end_batch' signal, gen_loss (unused) is accepeccted."
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