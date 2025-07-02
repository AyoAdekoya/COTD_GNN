import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from typing import Tuple, Dict

from create_nx import parse_verilog_netlist, training_filepairs, testing_filepairs
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                                                 #
#                   CONVERTING NXGRAPH TO PYG                     #
#                                                                 #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#Reading in all training and testing files
 #1. Create list of gatetypes
types_set = set()
Nx_Graphs = []
typeToindex = {}

i = 0
for filepath, reffilepath in training_filepairs + testing_filepairs:
    G_tmp = parse_verilog_netlist(filepath, reffilepath)
    Nx_Graphs.append(G_tmp)
    print("NX_GRAPH", i)
    print(G_tmp)
    print()
    i+=1
    # data=True allows us to access the attrs
    for node, attrs in G_tmp.nodes(data=True):
        # avoid duplication then convert back to list instead
        types_set.add(attrs['type'])

types_list = sorted(types_set)

# 2. Map type to index
for index, type in enumerate(types_list):
    typeToindex[type] = index


def graph_to_pyg_data(G: nx.DiGraph, typeToindex: Dict) -> Tuple[Data, Dict]:
    #used later to map nodes to an index and vice versa
    nodeToindex = {}
    indexTonode = {}

    #3. Create tensors for node features and labels
    X = []
    y = []

    for node, attrs in G.nodes(data=True):
        vector = torch.zeros(len(types_list), dtype=torch.float)
        type_index = typeToindex[attrs['type']]
        vector[type_index] = 1.0
        X.append(vector)
 
        label_str = ''
        if 'label' in attrs:
            label_str = attrs['label']
        else:
            label_str = 'Not_Trojaned'

        # label trojaned gates as 1 and nontrojaned as 0
        if label_str == 'Trojaned':
            y.append(1)
        else:
            y.append(0)
    
    # create a 2D tensor of each  stacked on rows not columns
    X_tensor = torch.stack(X, dim=0)
    # create a 1D tensor of labels
    y_tensor = torch.tensor(y, dtype=torch.long)

    #4. Create edges suitable for PyG
    edge_list = []
    #map gates to a number instead of str
    for index, node in enumerate(G.nodes()):
        nodeToindex[node] = index

    #reverse mapping for when we want to get the gates back from index
    for node, index in nodeToindex.items():
        indexTonode[index] = node

    for source, dest in G.edges():
        edge_list.append([nodeToindex[source], nodeToindex[dest]])
    edge_tensor = torch.tensor(edge_list, dtype=torch.long)
    #transpose the edges because PyG expects the edges to be of
    #the format source nodes on top row and dest bottom row
    edge_index = edge_tensor.t().contiguous()

    #message passing for GNN needs backwards edge so just flip the rows
    back_edge_index = edge_index.flip(0)
    bidir_edge_index = torch.cat([edge_index, back_edge_index], dim=1)

    data = Data(x=X_tensor, edge_index=bidir_edge_index, y=y_tensor)

    return data, indexTonode

# train_data = graph_to_pyg_data(G, typeToindex)

# print(train_data)

# g = to_networkx(train_data, to_undirected=True)
# nx.draw(g, with_labels=True, node_size=300, font_size=8)
# plt.show()

if __name__=='__main__':
    # simple smoke test
    from create_nx import parse_verilog_netlist, training_filepairs
    # parse one graph
    G = parse_verilog_netlist(*testing_filepairs[0])
    # convert to PyG Data
    data, indexTonode = graph_to_pyg_data(G, typeToindex)

    G_nx = to_networkx(data, to_undirected=True, node_attrs=['x','y']) 
    for name, attrs in G_nx.nodes(data=True):
        # attrs['x'] will be the feature vector,
        # attrs['y'] the label (if you passed them through)
        print(f"{indexTonode[name]}: {attrs}")

    print(data)