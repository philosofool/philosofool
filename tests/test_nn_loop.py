
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from philosofool.torch.callbacks import EndOnBatchCallback
from philosofool.torch.experimental.nn_loop import GANLoop
from philosofool.torch.nn_loop import (
    TrainingLoop,
    Publisher

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
        loop.publish(f"{loop.name}_control", 'end_fit')

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
                for expected_key in ['batch', 'loss', 'y_hat', 'y_true']:
                    if expected_key not in kwargs:
                        self.messages.append(f"missing keyword {expected_key}.")
                        return
                self.messages.append('batch_end')

            def on_epoch_end(self, loop, **kwargs):
                for expected_key in ['test_loss', 'y_hat', 'y_true']:
                    if expected_key not in kwargs:
                        self.messages.append(f"missing keyword {expected_key}.")
                        return
                self.messages.append('epoch_end')

            def on_fit_end(self, loop, **kwargs):
                self.messages.append('fit_end')

        callback = TestCallback()
        training_loop.fit(data_loader, data_loader, epochs=1, callbacks=[callback])
        expected = ['fit_start', 'epoch_start', 'batch_end', 'epoch_end', 'fit_end']
        assert len(callback.messages) <= len(expected), "An event was published more than expected."
        assert len(callback.messages) >= len(expected), "An expected event was not published."
        assert callback.messages == expected, "Some callback received the wrong keywords."


    def test_test(self, training_loop, data_loader):
        loss_value, y_hat, y = training_loop.test(data_loader)
        assert training_loop.model.training == False, "The model should be set to training."
        assert y_hat.requires_grad == False, "The results tensors should be deteched."
        assert y.requires_grad == False, "The results tensors should be deteched."
        assert y_hat.shape == y.shape, "These should be the same shape."
        assert torch.all(y == torch.tensor([[1., 0.], [0., 1.]])), "y values should match labels in data."

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
        for batch, loss_value, y_hat, y_true in loop_iterator:
            assert training_loop.model.training == True, "Training the model should set the model training to True."
            assert type(batch) is int
            assert type(loss_value) is float
            assert last_loss > loss_value, "The model loop should update the gradients to reduce to the loss."

            assert y_hat.shape == y_true.shape, "The shape of the predictions and the labels should be the same."
            assert y_hat.requires_grad == False
            assert torch.all((y_true == 1) + (y_true == 0)), "The values in y_true should be 1 or 0."

            # update to test loss on next iteration.
            last_loss = loss_value



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

        images = torch.rand((8, 3, 64, 64)) / 2 + .5
        generator = loop.generator
        fakes = generator(torch.randn(8, generator.input_size, 1, 1))
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

    def test_train(self, gan_loop: GANLoop):
        images, fakes, gen_params_initial, dis_params_initial = self._make_images_fakes(gan_loop)

        generator = gan_loop.generator
        discriminator = gan_loop.discriminator

        loader = DataLoader(TensorDataset(images), images.shape[0] // 4)
        n_iterations = 0

        for losses in gan_loop.train(loader):
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
        end_on_batch = EndOnBatchCallback(0)
        dataset = TensorDataset(images)
        data_loader = DataLoader(dataset, 2)
        gan_loop.fit(data_loader, epochs=2, callbacks=[end_on_batch, counter])
        assert counter.batches == 2
        assert counter.epochs == 2

    def test_fit__handles_end_fit(self, gan_loop: GANLoop):
        images, fakes, _, __ = self._make_images_fakes(gan_loop)
        counter = CountEpochsCallback()
        end_after_epoch = EndAfterOneEpoch()
        dataset = TensorDataset(images)
        data_loader = DataLoader(dataset, 2)
        gan_loop.fit(data_loader, epochs=2, callbacks=[end_after_epoch, counter])
        assert counter.batches == 4
        assert counter.epochs == 1
