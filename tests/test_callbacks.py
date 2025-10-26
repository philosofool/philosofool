from collections.abc import Callable
from collections import namedtuple
from tempfile import TemporaryDirectory
import os
import json

import matplotlib.pyplot as plt
import torch
from torch import nn
from philosofool.torch.callbacks import (
    EarlyStoppingCallabck, EndOnBatchCallback, HistoryCallback, JSONLoggerCallback, SnapshotCallback, VerboseTrainingCallback
)
from philosofool.torch.experimental.nn_loop import GANLoop
from philosofool.torch.nn_loop import (
    Publisher
)
from philosofool.torch.nn_models import Discriminator, Generator

class MessagesListener:
    """A test callback that records which events were received and their payloads."""
    LastCallArgs = namedtuple('LastCallArgs', ['args', 'kwargs'])

    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []  # [(event_name, args, kwargs), ...]
        self.handlers: dict[str, Callable]= {}  # caches handler functions for dynamic attributes

    def __getattr__(self, name: str) -> Callable:
        """Dynamically create a handler for any event (e.g., on_batch_end)."""
        if not name.startswith("on_"):
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")
        if name not in self.handlers:
            def handler(*args, **kwargs):
                self.calls.append((name, args, kwargs))
            self.handlers[name] = handler
        return self.handlers[name]

    def called(self, event_name: str) -> bool:
        """Return True if a given event was observed."""
        return any(name == event_name for name, *_ in self.calls)

    def last_call(self, event_name: str) -> tuple[tuple, dict] | None:
        """Return the most recent (args, kwargs) for a given event."""
        for name, args, kwargs in reversed(self.calls):
            if name == event_name:
                return self.LastCallArgs(args, kwargs)
        return None



def test_history_callback(training_loop):
    publisher = Publisher()
    callback = HistoryCallback()
    y_hat, y_hat_val = torch.tensor([.9, .1]), torch.tensor([.9, .1])
    y_true, y_true_val = torch.tensor([1, 0]), torch.tensor([1, 0])
    publisher.subscribe(training_loop.name, callback)
    publisher.publish(training_loop.name, 'epoch_start', training_loop)
    publisher.publish(training_loop.name, 'batch_end', training_loop, batch=1, loss=.05, y_hat=y_hat, y_true=y_true)
    publisher.publish(training_loop.name, 'epoch_end', training_loop, test_loss=.1, y_hat=y_hat_val, y_true=y_true_val)
    assert callback.history == {'loss': [.05], 'test_loss': [.1]}

    publisher.publish(training_loop.name, 'epoch_start', training_loop)
    publisher.publish(training_loop.name, 'batch_end', training_loop, batch=1, loss=.04, y_hat=y_hat, y_true=y_true)
    publisher.publish(training_loop.name, 'epoch_end', training_loop, test_loss=.09, y_hat=y_hat_val, y_true=y_true_val)
    assert callback.history == {'loss': [.05, .04], 'test_loss': [.1, .09]}, "HistoryCallback should accumulate new losses."

def test_history_callback__handles_GANLoop_kwargs(training_loop):
    callback = HistoryCallback()
    publisher = Publisher()

    publisher.subscribe(training_loop.name, callback)
    publisher.publish(training_loop.name, 'epoch_start', training_loop)
    publisher.publish(training_loop.name, 'batch_end', training_loop, batch=1, gen_loss=.05, dis_loss=.5)
    publisher.publish(training_loop.name, 'epoch_end', training_loop, epoch=1)
    assert callback.history == {'gen_loss': [.05], 'dis_loss': [.5]}


def test_history_callback__captures_metrics(training_loop):
    publisher = Publisher()
    callback = HistoryCallback()
    publisher.subscribe('training_loop', callback)

    publisher.publish('training_loop', 'metrics', training_loop, {'accuracy': .8, 'accuracy_val': .75})
    assert callback.history == {'accuracy': [.8], 'accuracy_val': [.75]}
    publisher.publish('training_loop', 'metrics', training_loop, {'accuracy': .78, 'accuracy_val': .68})
    assert callback.history == {'accuracy': [.8, .78], 'accuracy_val': [.75, .68]}


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
    assert listener.called('on_end_epoch')

def test_snapshot_callback(monkeypatch):
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

    def mockshow():
        return

    monkeypatch.setattr(plt, 'show', mockshow)

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
        callback = EarlyStoppingCallabck(patience=2, monitor='test_loss')
        training_loop.add_callbacks(callback)

        listener = MessagesListener()
        control_listener = MessagesListener()
        training_loop.subscribe('training_loop_control', control_listener)
        training_loop.subscribe('training_loop', listener)

        y = torch.tensor([1, 1])
        y_hat = torch.tensor([.5, .5])
        loss = torch.nn.functional.nll_loss(y_hat, y).mean().item()

        training_loop.publish(training_loop.name, 'epoch_end', test_loss=loss)
        training_loop.publish(training_loop.name, 'epoch_end', test_loss=loss)
        assert not control_listener.called('on_end_fit'), \
            "Loop should not emit end fit until patience number of turns without improvement. The first iteration alway improves."

        training_loop.publish(training_loop.name, 'epoch_end', test_loss=loss)
        assert control_listener.called('on_end_fit'), "The loops should stop after two iterations with no improvement."

    def test_resets_after_improvement(self, training_loop):
        callback = EarlyStoppingCallabck(patience=1, monitor='test_loss')
        training_loop.add_callbacks(callback)

        listener = MessagesListener()
        training_loop.subscribe('training_loop_control', listener)

        y = torch.tensor([1, 1])
        y_hat = torch.tensor([.5, .5])
        loss = torch.nn.functional.nll_loss(y_hat, y).mean().item()

        training_loop.publish(training_loop.name, 'epoch_end', test_loss=loss)
        training_loop.publish(training_loop.name, 'epoch_end', test_loss=loss - 1)
        training_loop.publish(training_loop.name, 'epoch_end', test_loss=loss)
        assert listener.called('on_end_fit'), "For patience == 1, loops should end after one step with no improvement."

        assert not listener.last_call('on_end_fit').kwargs, "kwargs should be empty."

    def test_on_fit_start(self, training_loop):
        callback = EarlyStoppingCallabck(patience=1, monitor='test_loss')
        training_loop.add_callbacks(callback)

        training_loop.publish(training_loop.name, 'epoch_end', test_loss=1)
        training_loop.publish(training_loop.name, 'epoch_end', test_loss=1)

        listener = MessagesListener()
        training_loop.subscribe('training_loop_control', listener)

        training_loop.publish(training_loop.name, 'fit_start')
        training_loop.publish(training_loop.name, 'epoch_end', test_loss=1)
        assert not listener.called('on_end_fit')
        # training_

    def test_on_fit_start__no_refresh(self, training_loop):
        callback = EarlyStoppingCallabck(patience=1, monitor='test_loss', refresh_on_new_fit=False)
        training_loop.add_callbacks(callback)

        training_loop.publish(training_loop.name, 'epoch_end', test_loss=1)
        training_loop.publish(training_loop.name, 'epoch_end', test_loss=1)

        listener = MessagesListener()
        training_loop.subscribe('training_loop_control', listener)

        training_loop.publish(training_loop.name, 'fit_start')
        training_loop.publish(training_loop.name, 'epoch_end', test_loss=1)
        assert listener.called('on_end_fit')



class TestMetricsCallback():
    def test_metrics_compute(self, training_loop):
        from philosofool.torch.metrics import Accuracy
        from philosofool.torch.callbacks import MetricsCallback
        accuracy = Accuracy('binary')
        callback = MetricsCallback([accuracy])

        callback.on_batch_end(training_loop, loss=.1, y_hat=torch.tensor([1., 1., 0., 0]), y_true=torch.tensor([1., 0., 1., 0]))
        callback.on_batch_end(training_loop, loss=.1, y_hat=torch.tensor([1., 1., 0., 0]), y_true=torch.tensor([1., 1., 0., 0]))

        assert accuracy.correct == 6
        assert accuracy.count == 8
        callback.on_epoch_end(training_loop, y_hat=torch.tensor([1., 1., 0., 0]), y_true=torch.tensor([1., 1., 0., 0]))
        assert accuracy.count == 0., "Count should reset each epoch."
        assert accuracy.correct == 0., "Correct should reset each epoch"

    def test_metrics_publish(self, training_loop):
        from philosofool.torch.metrics import Accuracy
        from philosofool.torch.callbacks import MetricsCallback
        from philosofool.torch.nn_loop import TrainingLoop
        assert isinstance(training_loop, TrainingLoop)

        accuracy = Accuracy('binary')
        callback = MetricsCallback([accuracy])
        training_loop.add_callbacks(callback)

        listener = MessagesListener()
        training_loop.subscribe('training_loop', listener)

        training_loop.publish('training_loop', 'batch_end', batch=0,  loss=.5, y_hat=torch.tensor([1., 1., 0., 0]), y_true=torch.tensor([1., 0., 1., 0]))
        training_loop.publish('training_loop', 'batch_end', batch=1,  loss=.4, y_hat=torch.tensor([1., 1., 0., 0]), y_true=torch.tensor([1., 1., 0., 0]))
        training_loop.publish('training_loop', 'epoch_end',  test_loss=.3, y_hat=torch.tensor([1., 1., 0., 0]), y_true=torch.tensor([1., 1., 0., 0]))

        assert listener.called('on_metrics')
        metrics = listener.last_call('on_metrics').kwargs.get('metrics')
        assert metrics is not None, "Metrics should be in the call kwargs."
        assert 'accuracy' in metrics
        assert 'accuracy_val' in metrics

    def test_captures_losses(self, training_loop):
        from philosofool.torch.metrics import Accuracy
        from philosofool.torch.callbacks import MetricsCallback
        from philosofool.torch.nn_loop import TrainingLoop
        assert isinstance(training_loop, TrainingLoop)

        callback = MetricsCallback([])
        training_loop.add_callbacks(callback)

        listener = MessagesListener()
        training_loop.subscribe('training_loop', listener)

        training_loop.publish('training_loop', 'batch_end', batch=0,  loss=.5, y_hat=torch.tensor([1., 1., 0., 0]), y_true=torch.tensor([1., 0., 1., 0]))
        training_loop.publish('training_loop', 'batch_end', batch=1,  loss=.4, y_hat=torch.tensor([1., 1., 0., 0]), y_true=torch.tensor([1., 1., 0., 0]))
        training_loop.publish('training_loop', 'epoch_end',  test_loss=.3, y_hat=torch.tensor([1., 1., 0., 0]), y_true=torch.tensor([1., 1., 0., 0]))

        assert listener.called('on_metrics')
        metrics = listener.last_call('on_metrics').kwargs.get('metrics')
        assert metrics is not None, "Metrics should be in the call kwargs."
        assert metrics.get('loss_val') == .3
        loss = metrics.get('loss')
        assert loss == .45
