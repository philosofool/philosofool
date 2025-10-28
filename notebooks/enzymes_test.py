# %%
from collections.abc import Callable
from typing import Any
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric import nn as gnn
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.loader import DataLoader

from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


# %%
class ConfusionMatrix:
    def __init__(self):
        self.confusion_matrix = np.empty(0)
        self._y_hat = []
        self._y_true = []

    def update(self, y_hat, y_true):
        self._y_hat.append(np.asarray(y_hat))
        self._y_true.append(np.asarray(y_true))

    def compute(self):
        y_hat = np.concat(self._y_hat)
        y_true = np.concat(self._y_true)
        pred = np.argmax(y_hat, axis=1)
        confusion_matrix_ = confusion_matrix(y_true, pred)
        return confusion_matrix_

    def reset(self):
        self._y_true = []
        self._y_hat = []

# %%
def test_confusion_matrix():
    matrix = ConfusionMatrix()
    matrix.update(torch.tensor([[.1, .9], [.9, .1]]), torch.tensor([0, 1]))
    matrix.update(torch.tensor(
        [
            [.9, .1],
            [.9, .1],
            [.1, .9],
        ]),
        torch.tensor([0, 1, 1])
        )

    result = matrix.compute()
    assert result[0, 0] == 1
    assert result[1, 1] == 1

test_confusion_matrix()

# NOTE: dataset expects this capitalization of the path and names.
enzyme_data = TUDataset(root='local_data/ENZYMES', name='ENZYMES')
print(enzyme_data, 'num features: ', enzyme_data.num_node_features, 'num classes: ', enzyme_data.num_classes)

# The data includes 600 graphs.
# each graph has a a single taget variable, which is one of the six classes; y=[1]
# the graphs themselves have variables numbers of nodes and edges.
print(enzyme_data[0])
print(enzyme_data[1])

# %%
class GraphPredGCN(torch.nn.Module):
    def __init__(self, node_features:int, num_classes: int, hidden_channels: int, n_convolutions: int, dropout: float):
        super().__init__()
        self.conv_layers = []
        for i in range(n_convolutions):
            if i == 0:
                conv = GCNConv(node_features, hidden_channels)
            else:
                conv = GCNConv(hidden_channels, hidden_channels)
            # add by name to make a model parameter.
            setattr(self, f'conv_{i}', conv)
            self.conv_layers.append(conv)

        self.linear = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        pool_step = 20
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            x = F.relu(x)
            if i > 0 and not i % pool_step:
                x = gnn.pool.TopKPooling(x, batch)
            if self.dropout:
                x = nn.Dropout(self.dropout)(x)

        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x

# %%
enzyme_data = enzyme_data.shuffle()
train_size = 400
train_enzymes = enzyme_data[:train_size]
test_enzymes = enzyme_data[train_size:]

def graph_dataset_summary(dataset):
    n = len(dataset)
    n_classes = dataset.num_classes
    n_features = dataset.num_node_features
    label_dis = F.one_hot(dataset.y).sum(dim=0) if hasattr(dataset, 'y') else None
    report = f"Dataset with {n} nodes, {n_classes} classes and {n_features} node features."
    if label_dis is not None:
        label_dis = ', '.join([f'{x:.3f}' for x in label_dis / n])
        report += f"\nDistribution of classes is {label_dis}."
    print(report)

graph_dataset_summary(train_enzymes)

# %%
from collections.abc import Iterable

from philosofool.torch.callbacks import VerboseTrainingCallback
from philosofool.torch.nn_loop import TrainingLoop

class DefaultGraphAdapter:
    def __init__(self, features: Iterable[str], target='y'):
        self.features = features
        self.target = target

    def get_inputs(self, data_batch) -> list:
        inputs = [getattr(data_batch, feature) for feature in self.features]
        return inputs

    def get_target(self, data_batch) -> torch.Tensor:
        return getattr(data_batch, self.target)

    def get_train_mask(self, data_batch):
        return getattr(data_batch, 'train_mask', None)

    def get_val_mask(self, data_batch):
        return getattr(data_batch, 'test_mask', None)


class GraphTrainingLoop(TrainingLoop):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss: 'Loss', adapter: DefaultGraphAdapter, name='graph_training_loop'):
        super().__init__(model, optimizer, loss, adapter, name)
        if self.adapter is None:
            self.adapter = DefaultGraphAdapter()


    def process_batches(self, loader: DataLoader, with_grad: bool = True, eval: bool | None = None):
        self._set_train_state(with_grad, eval)

        for batch, data in enumerate(loader):
            X = self.adapter.get_inputs(data)
            y = self.adapter.get_target(data)
            if with_grad:
                # !! return signature breaks contract of parent class!
                # we thinking throught how to get this work without a child
                # class. This will be a pain point.
                loss, pred_masked, y_masked = self._compute_loss_pred_optimize(X, y)
            else:
                loss, pred_masked, y_masked = self._compute_loss_pred(X, y)
            yield batch, loss.item(), pred_masked.detach(), y_masked

    def _compute_loss_pred_optimize(self, X, y):
        mask = self.adapter.get_train_mask(X)
        pred = self.model(*X)
        if mask is None:
            # in this case, treat pred_masked is pred
            pred_masked = pred
            y_masked = y
        else:
            pred_masked = pred[mask]
            y_masked = y[mask]
        loss = self.loss(pred_masked, y_masked)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss, pred_masked, y_masked

    def _compute_loss_pred(self, X, y):
        mask = self.adapter.get_train_mask(X)
        with torch.no_grad():
            pred = self.model(*X)
            if mask is None:
                # in this case, treat pred_masked is pred
                pred_masked = pred
                y_masked = y
            else:
                pred_masked = pred[mask]
                y_masked = y[mask]
            loss = self.loss(pred_masked, y_masked)
        return loss, pred_masked, y_masked


    def test(self, loader: DataLoader):
        self.model.eval()
        test_loss, correct = 0., 0.
        with torch.no_grad():
            for batch, data in enumerate(loader):
                pred = self.model(data.x, data.edge_index, data.batch)
                test_loss += self.loss(pred, data.y).item()
                correct += (pred.argmax(1) == data.y).sum().item()
        test_loss /= len(loader)
        correct /= len(loader.dataset)
        return correct, test_loss


# %%
import seaborn as sns

def plot_history(history: dict, plots: list):
    fig, ax = plt.subplots(1, len(plots))
    fig.set_size_inches(len(plots) * 4, 4)
    plots = [(idx, key + suffix) for suffix in ['', '_val'] for idx, key in enumerate(plots)]
    fig.tight_layout()
    for idx, key in plots:
        value = history[key]
        ax[idx].plot(range(len(value)), value, label=key)

def heatmaps(history):
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(8, 4)
    fig.tight_layout()
    sns.heatmap(history['confusionmatrix'][-1], ax=ax[0])
    sns.heatmap(history['confusionmatrix_val'][-1], ax=ax[1])


# %%
def pooling_experiment():
    ...


# %%
from matplotlib.pyplot import hist
from philosofool.torch.callbacks import MetricsCallback
from philosofool.torch.metrics import Accuracy

try:
    histories
except NameError:
    histories = []

train_loader = DataLoader(train_enzymes, batch_size=32)
test_loader = DataLoader(test_enzymes, batch_size=32)

model = GraphPredGCN(train_enzymes.num_node_features, train_enzymes.num_classes, 16, 2, dropout=0)

optimizer = torch.optim.Adam(model.parameters(), lr=.005)
# optimizer = torch.optim.SGD(model.parameters(), momentum=.9, lr=.01)
confusion_callback = ConfusionMatrix()
loop = GraphTrainingLoop(model, optimizer, torch.nn.CrossEntropyLoss(), adapter=DefaultGraphAdapter(['x', 'edge_index', 'batch']))
loop.fit(train_loader, test_loader, epochs=300, callbacks=[
    MetricsCallback([Accuracy('multiclass'), confusion_callback])])
histories.append(loop.history)
# loop.history


# %%
plot_history(loop.history, ['loss', 'accuracy'])
loop.history['confusionmatrix_val'][-1],

# %%
heatmaps(loop.history)

# %%
loop.history


# %%
one_batch = next(iter(train_loader))
one_batch.to('cpu')
