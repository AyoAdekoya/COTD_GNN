import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, BatchNorm
from CSVtoPyg import create_all_data
from zScore import compute_scoap_stats
import random


def save_datasets(train_list, test_list, save_dir="saved_datasets"):
    import os
    os.makedirs(save_dir, exist_ok=True)
    torch.save(train_list, f"{save_dir}/train_dataset.pt")
    torch.save(test_list, f"{save_dir}/test_dataset.pt")
    print(f"Datasets saved in {save_dir}/")

def load_datasets(save_dir="saved_datasets"):
    train_list = torch.load(f"{save_dir}/train_dataset.pt", weights_only=False)
    test_list  = torch.load(f"{save_dir}/test_dataset.pt", weights_only=False)
    return train_list, test_list

def get_loaders_from_saved(batch_size=1, shuffle_train=True, save_dir="saved_datasets"):
    train_list, test_list = load_datasets(save_dir)
    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=shuffle_train)
    test_loader  = DataLoader(test_list, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_loaders(batch_size=1, shuffle_train=True):
    compute_scoap_stats(folder_path="Trojan_GNN/Trust-Hub/Scoap_Scores")

    # Get the data lists
    trojan_list, free_list = create_all_data()

    # Remove None entries
    trojan_list = [d for d in trojan_list if d is not None]
    free_list   = [d for d in free_list if d is not None]

    # Shuffle
    random.shuffle(trojan_list)
    random.shuffle(free_list)

    # Split 80/20
    def split_80_20(lst):
        split_idx = int(len(lst) * 0.8)
        return lst[:split_idx], lst[split_idx:]

    trojan_train, trojan_test = split_80_20(trojan_list)
    free_train,   free_test   = split_80_20(free_list)

    # Combine
    train_dataset = trojan_train + free_train
    test_dataset  = trojan_test  + free_test

    # Final shuffle
    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train():
    train_loader, test_loader = get_loaders_from_saved()

    print("Training model...")

    hidden_dim  = 64
    num_classes = 2  # binary classification

    # Get first batch to determine input dimension
    first_batch = next(iter(train_loader))
    in_dim = first_batch.num_node_features

    # Now create model
    model = TrojanGNN(in_dim, hidden_dim, num_classes)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 100
    model.train()

    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch.x
            edge_index = batch.edge_index

            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
            out = model(x, edge_index)

            # Compute loss
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

            out = model(x, edge_index)
            preds = out.argmax(dim=1)

            correct_batch = (preds == batch.y).sum().item()
            total_batch = batch.y.size(0)

            correct += correct_batch
            total += total_batch

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
    # Run this part to save datasets as a file #########################

    # compute_scoap_stats()
    # trojan_list, free_list = create_all_data()

    # # Remove None
    # trojan_list = [d for d in trojan_list if d is not None]
    # free_list = [d for d in free_list if d is not None]

    # # Split
    # def split_80_20(lst):
    #     idx = int(len(lst) * 0.8)
    #     return lst[:idx], lst[idx:]

    # tj_train, tj_test = split_80_20(trojan_list)
    # fr_train, fr_test = split_80_20(free_list)

    # train_dataset = tj_train + fr_train
    # test_dataset  = tj_test  + fr_test

    # # Shuffle
    # import random
    # random.shuffle(train_dataset)
    # random.shuffle(test_dataset)

    # # Save to disk
    # save_datasets(train_dataset, test_dataset)

    ####################################################################
    # Run this part to retrieve the saved datasets
    train()