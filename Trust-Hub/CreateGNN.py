import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, BatchNorm
from CSVtoPyG import create_graph, nx_to_pyG, typeToindex, indexTonode


def train():
    # Placeholder: initialize your training and testing data loaders
    # Example:
    # train_dataset = [...]  # list of Data objects for training graphs
    # test_dataset  = [...]  # list of Data objects for testing graphs
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_loader  = DataLoader(test_dataset,  batch_size=32)
    train_loader = None  # TODO: replace with actual DataLoader
    test_loader  = None  # TODO: replace with actual DataLoader

    # Create your model
    in_dim      = None  # will be set once we inspect a batch
    hidden_dim  = 64
    num_classes = 2  # binary classification

    model = TrojanGNN(in_dim, hidden_dim, num_classes)

    # Use Adam optimizer and CrossEntropyLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 100
    model.train()

    for epoch in range(num_epochs):
        for batch in train_loader:
            # Ensure input dimension is set
            if in_dim is None:
                in_dim = batch.num_node_features
                model = TrojanGNN(in_dim, hidden_dim, num_classes)

            optimizer.zero_grad()
            x = batch.x
            edge_index = batch.edge_index

            # Forward pass
            out = model(x, edge_index)

            # Compute loss on training nodes
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Testing loop
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            x = batch.x
            edge_index = batch.edge_index

            # Forward pass
            out = model(x, edge_index)

            # Predictions
            preds = out.argmax(dim=1)

            # Update counts
            correct_batch = (preds == batch.y).sum().item()
            total_batch = batch.y.size(0)

            correct += correct_batch
            total += total_batch

    # Compute accuracy
    if total > 0:
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    else:
        print("No test data available to compute accuracy.")


class TrojanGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(TrojanGNN, self).__init__()
        # First graph convolution layer
        self.conv1 = GCNConv(in_dim, hidden_dim)
        # Batch normalization for hidden dimension
        self.bn1 = BatchNorm(hidden_dim)
        # Second graph convolution layer for output
        self.conv2 = GCNConv(hidden_dim, num_classes)
        # Dropout probability
        self.dropout_p = 0.5

    def forward(self, x, edge_index):
        # First convolution
        x1 = self.conv1(x, edge_index)
        # Apply batch normalization
        x2 = self.bn1(x1)
        # Apply ReLU activation
        x3 = F.relu(x2)
        # Apply dropout
        x4 = F.dropout(x3, p=self.dropout_p, training=self.training)
        # Second convolution
        x5 = self.conv2(x4, edge_index)
        return x5


if __name__ == '__main__':
    # Example of building a graph from files
    # netlist_path = 'design.v'
    # attrs_csv    = 'attrs.csv'
    # mapping_txt  = 'mapping.txt'
    # G = create_graph(netlist_path, attrs_csv, mapping_txt)
    # data = nx_to_pyG(G)
    
    # Now train the model
    train()