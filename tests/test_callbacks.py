from tempfile import TemporaryDirectory
import torch
from torch import nn
import os
import json
from philosofool.torch.callbacks import EndOnBatchCallback, HistoryCallback, JSONLoggerCallback, SnapshotCallback, VerboseTrainingCallback
from philosofool.torch.nn_loop import (
    GANLoop,
    Publisher
)
from philosofool.torch.nn_models import Discriminator, Generator


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