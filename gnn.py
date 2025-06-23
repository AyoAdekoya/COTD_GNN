# simple example of GNN for learning purposes

# import torch
# import torch.nn.functional as F
import networkx as nx
# from torch_geometric.utils import from_networkx
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data

# create graph using nx

G = nx.DiGraph()
G.add_edges_from([
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 4)
])

for node in G.nodes:
    G.nodes[node]['x'] = [1.0 * node]  # just a dummy feature (node index)

labels = {0: 1, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 1, 7:1}

for node, label in labels.items():
    G.nodes[node]['y'] = label

for node in G.nodes:
    print(G.nodes[node]['x'])
    print(G.nodes[node]['y'])