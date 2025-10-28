# just learning at the moment.

# see https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html

import torch
from torch_geometric.data import Data


# define and undirected graph: 4 edges for 2 connected nodes.
edge_index = torch.tensor(
    [[0, 1, 1, 2],
     [1, 0, 2, 1]],
    dtype=torch.long
)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

print("Graphs in torch.geometric are described by instances of torch_geometric.data.Data")
data = Data(x=x, edge_index=edge_index)

data.validate(raise_on_error=True)
print(data)
print(data.keys())
for key, item in data:
    print(f'key {key} in data with item {item}.')

print(data.num_nodes, data.num_edges, data.num_node_features)

# draw the netowrk:


# look at some data sets.
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

# NOTE: dataset expects this capitalization of the path and names.
enzyme_data = TUDataset(root='/workspace/local_data/ENZYMES', name='ENZYMES')
print(enzyme_data, 'num features: ', enzyme_data.num_node_features, 'num classes: ', enzyme_data.num_classes)

print(enzyme_data[0])
print(enzyme_data[1])

from torch_geometric.datasets import Planetoid
citation_data = Planetoid(root='/workspace/local_data/Cora', name='Cora')
loader = DataLoader(citation_data, batch_size=32)

for batch in loader:
    print(batch.batch)
    print(batch[0].x)
    print(batch[0].edge_index)
    break

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data: Data, *_,):
        """Compute the forward pass of the network.

        Parameters
        ----------
        data:
            Torch geometric Data instance of graph data.
        *_:
            An unused parameter, for compatibility with nn.Module subsclasses,
            which often expect data and, separately, labels.
        """
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x =  F.log_softmax(x, dim=1)
        return x

citation_graph = citation_data[0]

model = GCN(citation_data)
optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=5e-4)


print("Training network on citations.")
n_epoch = 500
best_loss = torch.inf
n_no_improvement = 0
for epoch in range(n_epoch):
    model.train()
    optimizer.zero_grad()
    pred = model(citation_graph)
    loss = F.nll_loss(pred[citation_graph.train_mask], citation_graph.y[citation_graph.train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    pred = model(citation_graph).argmax(dim=1)
    correct = (pred[citation_graph.test_mask] == citation_graph.y[citation_graph.test_mask]).sum()
    accuracy = int(correct) / int(citation_graph.test_mask.sum())
    if epoch % 5 == 0:
        print(f"Loss: {loss.item():.5f}, Accuracy: {accuracy:.3f}  (epoch {epoch})")
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_accuracy = accuracy
        n_no_improvement = 0
    else:
        n_no_improvement += 1
        if n_no_improvement > 100:
            break

print(f"\nLoss of best model: {best_loss:.5f}, accuracy {accuracy:.3f}")
