import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from nx_to_pyG import graph_to_pyg_data, Nx_Graphs, typeToindex
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                                                 #
#                   TRAINING vs. TESTING SETS                     #
#                                                                 #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#5. Create training and test graphs
train_data_list = []
indexTonode_list = []
for i in range(16):
    pyG_train_data, indexTonode = graph_to_pyg_data(Nx_Graphs[i], typeToindex)
    num_train_nodes = pyG_train_data.num_nodes
    print(pyG_train_data, num_train_nodes)
    assert num_train_nodes is not None
    #train on all nodes of the train graph
    pyG_train_data.train_mask = torch.ones(num_train_nodes, dtype=torch.bool)
    train_data_list.append(pyG_train_data)
    indexTonode_list.append(indexTonode)

test_data_list = []
for i in range(16,20):
    pyG_test_data, indexTonode = graph_to_pyg_data(Nx_Graphs[i], typeToindex)
    num_test_nodes = pyG_test_data.num_nodes
    print(pyG_test_data, num_test_nodes)
    assert num_test_nodes is not None
    #evaluate on all nodes of the test graph
    pyG_test_data.test_mask = torch.ones(num_test_nodes, dtype=torch.bool)
    test_data_list.append(pyG_test_data)
    indexTonode_list.append(indexTonode)

# batch_size 2 to start and then drop to 1 if needed
train_loader = DataLoader(train_data_list, batch_size=5, shuffle=True)
test_loader  = DataLoader(test_data_list,  batch_size=1, shuffle=False)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                                                 #
#                  GCN for Node Classification                    #
#                                                                 #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#6 Create the class for the GCN classification
class NodeClassifier(torch.nn.Module):
    def __init__(self, input_features, hidden_dim=64, num_classes=2):
        super(NodeClassifier, self).__init__()
        self.conv1 = GCNConv(input_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, num_classes)

    #message passing forward on step
    def forward(self, X, edge_index):
        X = self.conv1(X, edge_index)
        X = F.relu(X)
        X = self.conv2(X, edge_index)
        X = F.relu(X)
        X = self.lin(X)
        return X
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                                                                 #
#                           TRAINING LOOP                         #
#                                                                 #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def main():
    #6. Create a model based on the GCN object and train the model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = NodeClassifier(input_features=train_data.num_node_features).to(device)
    model = NodeClassifier(input_features=len(typeToindex))
    #using default learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for batch in train_loader:
            #reset the batch gradients
            optimizer.zero_grad()
            #forward pass
            output = model(batch.x, batch.edge_index)

        #possible fix to train_data.y being classified as int instead of Data
        # assert train_data.edge_index is not None
        # assert train_data.x is not None
        # train_data.x          = train_data.x.to(device)
        # train_data.edge_index = train_data.edge_index.to(device)
        # train_data.y          = train_data.y.to(device) # type: ignore
        # train_data.train_mask = train_data.train_mask.to(device)

            #calculating training loss
            assert isinstance(batch.y, torch.Tensor)
            train_logits = output[batch.train_mask]
            train_labels = batch.y[batch.train_mask]
            loss = F.cross_entropy(train_logits, train_labels)

            #backward + step
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}  Avg Loss {total_loss/len(train_loader):.4f}")

    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        # i = 16
        # for batch in test_loader:
        #     print("Design",i)
        #     print()
        #     out     = model(batch.x, batch.edge_index)
        #     preds   = out.argmax(dim=1)

        #     correct += (preds[batch.test_mask] == batch.y[batch.test_mask]).sum().item()
        #     total   += batch.test_mask.sum().item()

        #     # optional: print per-node   
        #     for idx in range(batch.num_nodes):
        #         true_lbl = 'Trojaned' if batch.y[idx] == 1 else 'Not_Trojaned'
        #         pred_lbl = 'Trojaned' if preds[idx]  == 1 else 'Not_Trojaned'
        #         print(f"{indexTonode_list[i][idx]:<8s} true={true_lbl:<14s} pred={pred_lbl}")
        #     i += 1

        batch = next(iter(test_loader))
        print("Design 16")
        print()
        out     = model(batch.x, batch.edge_index)
        preds   = out.argmax(dim=1)

        correct += (preds[batch.test_mask] == batch.y[batch.test_mask]).sum().item()
        total   += batch.test_mask.sum().item()

        #optional: print per-node   
        for idx in range(batch.num_nodes):
            true_lbl = 'Trojaned' if batch.y[idx] == 1 else 'Not_Trojaned'
            pred_lbl = 'Trojaned' if preds[idx]  == 1 else 'Not_Trojaned'
            print(f"{(indexTonode_list[16])[idx]:<8s} true={true_lbl:<14s} pred={pred_lbl}")


    print(f"Test accuracy: {correct}/{total} = {correct/total:.4f}") 

if __name__=='__main__':
    main()


# Switch up the training data set so that you aren't just always training
# on the same designs
    # Idea: have all the data sets in a list, and use a random method call to
    # choose what index should go into the train vs test list on all of the 
    # circuits.
    # Once we get a higher percent accuracy, we save that model somehow
# 
# Then, include some trojan_free desings into the training data