from tempfile import TemporaryDirectory
import torch
from torch import nn
import os
import json
from philosofool.torch.callbacks import (
    EarlyStoppingCallabck, EndOnBatchCallback, HistoryCallback, JSONLoggerCallback, SnapshotCallback, VerboseTrainingCallback
)
from philosofool.torch.nn_loop import (
    GANLoop,
    Publisher
)
from philosofool.torch.nn_models import Discriminator, Generator

class MessagesListener:
    """Register when a message is recieved."""
    def __init__(self):
        self.messages = {}

    def __getattr__(self, value):
        if isinstance(value, str) and value.startswith('on_'):
            self.messages[value] = True
        return self.handler

    def handler(self, *args, **kwargs):
        return None

def test_history_callback():
    publisher = Publisher()
    callback = HistoryCallback(batch_end=['loss'], epoch_end=['test_loss'])
    y_hat, y_hat_val = torch.tensor([.9, .1]), torch.tensor([.9, .1])
    y_true, y_true_val = torch.tensor([1, 0]), torch.tensor([1, 0])
    publisher.subscribe('event', callback)
    publisher.publish('event', 'epoch_start', None)
    publisher.publish('event', 'batch_end', None, batch=1, loss=.05, y_hat=y_hat, y_true=y_true)
    publisher.publish('event', 'epoch_end', None, test_loss=.1, y_hat=y_hat_val, y_true=y_true_val)
    assert callback.history == {'loss': [.05], 'test_loss': [.1]}


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
    # assert logs['test_accuracy'] == [1 / 2]  # THIS NEEDS IMPLEMENTING


def test_end_on_batch(training_loop):

    listener = MessagesListener()
    training_loop._publisher.subscribe('training_loop_control', listener)
    end_on_batch = EndOnBatchCallback(2)
    assert end_on_batch.on_batch_end(training_loop, batch=2, gen_loss=.1) is None, "Emitting signals is deprecated."
    assert end_on_batch.on_batch_end(training_loop, batch=1) is None, "Should not end on batch one. Missing gen_loss is accepted."
    assert listener.messages.get('on_end_epoch')

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

class TestEarlyStopping:
    def test_emits_end_fit(self, training_loop):
        """Assure stops after number of expected turns."""
        callback = EarlyStoppingCallabck(patience=2, monitor='val_loss')
        training_loop.add_callbacks(callback)
        # publisher = Publisher()

        class MessagesListener:
            def __init__(self):
                self.messages = {}

            def __getattr__(self, value):
                if isinstance(value, str) and value.startswith('on_'):
                    self.messages[value] = True
                return self.handler

            def handler(self, *args, **kwargs):
                return None

        listener = MessagesListener()
        training_loop._publisher.subscribe('training_loop_control', listener)

        y = torch.tensor([1, 1])
        y_hat = torch.tensor([.5, .5])
        loss = torch.nn.functional.nll_loss(y_hat, y).mean().item()

        for i in range(2):
            signals = training_loop.publish(training_loop.name, 'epoch_end', test_loss=loss)
            assert signals is None
        assert listener.messages.get('on_end_fit') is None
        signals = training_loop.publish(training_loop.name, 'epoch_end', test_loss=loss)
        assert not signals, "Emitting signals is deprecated."
        assert listener.messages.get('on_end_fit')

    def test_resets_after_improvement(self, training_loop):
        callback = EarlyStoppingCallabck(patience=1, monitor='val_loss')
        training_loop.add_callbacks(callback)

        listener = MessagesListener()
        training_loop._publisher.subscribe('training_loop_control', listener)

        y = torch.tensor([1, 1])
        y_hat = torch.tensor([.5, .5])
        loss = torch.nn.functional.nll_loss(y_hat, y).mean().item()

        training_loop.publish(training_loop.name, 'epoch_end', test_loss=loss)
        training_loop.publish(training_loop.name, 'epoch_end', test_loss=loss - 1)
        assert listener.messages.get('on_end_fit') is None

        training_loop.publish(training_loop.name, 'epoch_end', test_loss=loss)
        training_loop.publish(training_loop.name, 'epoch_end', test_loss=loss)
        assert listener.messages.get('on_end_fit')
