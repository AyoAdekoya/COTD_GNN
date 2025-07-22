import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, BatchNorm, SAGEConv
from CSVtoPyg import create_all_data
from zScore import compute_scoap_stats
import random

train_dataset = []
test_dataset = []
# val_dataset = []

def save_datasets(train_list, test_list, save_dir="saved_datasets"):
    import os
    os.makedirs(save_dir, exist_ok=True)
    torch.save(train_list, f"{save_dir}/contesttrain_dataset.pt")
    torch.save(test_list, f"{save_dir}/contesttest_dataset.pt")
    # torch.save(val_list, f"{save_dir}/val_dataset.pt")
    print(f"Datasets saved in {save_dir}/")

def load_datasets(save_dir="saved_datasets"):
    train_list = torch.load(f"{save_dir}/contesttrain_dataset.pt", weights_only=False)
    test_list  = torch.load(f"{save_dir}/contesttest_dataset.pt", weights_only=False)
    # val_list  = torch.load(f"{save_dir}/val_dataset.pt", weights_only=False)
    return train_list, test_list

def get_loaders_from_saved(batch_size=1, shuffle_train=True, save_dir="saved_datasets"):
    # this is technically saving trojan and free not train and test
    train_dataset, test_dataset = load_datasets(save_dir)
    # training = list(train_dataset)
    # train = training[:-2]
    # trojaned_list = train[:100]

    total = train_dataset + test_dataset
    random.shuffle(total)
    # freed_list = [free_list[i] for i in range (12)]

    # Split 80/20
    def split_80_20(lst):
        split_idx = int(len(lst) * 0.8)
        return lst[:split_idx], lst[split_idx:]
    
    trojan_train, trojan_test = split_80_20(total)
    # free_train,   free_test   = split_80_20(freed_list)
   
    # # Combine
    train_dataset = trojan_train
    test_dataset  = trojan_test 

    print("Total graphs: ", len(train_dataset) + len(test_dataset))
    graph_weights = []
    for data in train_dataset:
        pos = int((data.y==1).sum().item())
        graph_weights.append(pos + 0.1)  # add a small epsilon so 0‑positive graphs still get sampled

    sampler = WeightedRandomSampler(weights=graph_weights,num_samples=len(graph_weights),
                                    replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,sampler=sampler)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=shuffle_train)
    return train_loader, test_loader


def get_loaders(batch_size=1, shuffle_train=True):
    compute_scoap_stats(folder_path="Trojan_GNN/Trust-Hub/Scoap_Scores")

    # Get the data lists
    trojan_list, free_list = create_all_data()

    # Remove None entries
    trojan_list = [d for d in trojan_list if d is not None]
    free_list   = [d for d in free_list if d is not None]

    # trojaned_list = [trojan_list[i] for i in range(28)]
    # freed_list = [free_list[i] for i in range (12)]

    # n = len(trojan_list)
    # split = int(n * 0.8)
    print("Free graphs:", len(free_list))
    # trojaned_list = [i for i in (trojan_list[:split])]
    # freed_list = [i for i in (trojan_list[split:])]

    # Shuffle
    random.shuffle(trojan_list)
    random.shuffle(free_list)

    save_datasets(trojan_list, free_list)

    # Split 80/20
    def split_80_20(lst):
        split_idx = int(len(lst) * 0.8)
        return lst[:split_idx], lst[split_idx:]
    # def split_80_10_10(lst):
    #     n = len(lst)
    #     split1 = int(n * 0.8)
    #     split2 = int(n * 0.9)  # 80% + 10%
        # return lst[:split1], lst[split1:split2], lst[split2:]

    trojan_train, trojan_test = split_80_20(trojan_list)
    free_train,   free_test   = split_80_20(free_list)
    # trojan_train, val_test, trojan_test = split_80_10_10(trojan_list)
    # free_train, free_val_test, free_test   = split_80_10_10(free_list)

    # Combine
    train_dataset = trojan_train + free_train
    test_dataset  = trojan_test  + free_test

    print("Total graphs: ", len(train_dataset) + len(test_dataset))

# Possibly only testing on some nodes maybe
    # for d in train_dataset:
    #     num = d.num_nodes
    #     d.train_mask = torch.ones(num, dtype=torch.bool)
    # for d in test_dataset:
    #     num = d.num_nodes
    #     d.test_mask = torch.ones(num, dtype=torch.bool)
    # val_dataset = val_test + free_val_test

    # Final shuffle
    random.shuffle(train_dataset)
    # random.shuffle(val_dataset)
    random.shuffle(test_dataset)

    # DataLoaders
    # graph_weights = []
    # for data in train_dataset:
    #     pos = int((data.y==1).sum().item())
    #     graph_weights.append(pos + 0.2)  # add a small epsilon so 0‑positive graphs still get sampled

    # sampler = WeightedRandomSampler(weights=graph_weights,num_samples=len(graph_weights),
    #                                 replacement=True)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size,sampler=sampler)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_train)

    return train_loader, test_loader

def train():
    train_loader, test_loader = get_loaders_from_saved()
    # train_loader, test_loader = get_loaders()

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
    # criterion = torch.nn.CrossEntropyLoss()

    all_y = torch.cat([d.y for d in train_loader], dim=0)
    num_neg = (all_y==0).sum().item()
    num_pos = (all_y==1).sum().item()
    class_weights = torch.tensor([1, num_neg/num_pos], dtype=torch.float)

    # criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    # criterion = FocalLoss(weight=class_weights)
    criterion = FocalLoss(weight=torch.tensor([1,1], dtype=torch.float))


    # Training loop
    num_epochs = 10
    model.train()

    for epoch in range(num_epochs):
        for batch in train_loader:
            print(f"Training with {batch.design_name}.")
            optimizer.zero_grad()
            x = batch.x
            edge_index = batch.edge_index

            # x = torch.nan_to_num(x, nan=normalize_score(10000, "co"), posinf=0.0, neginf=0.0)
            if torch.isnan(batch.x).any() or torch.isinf(batch.x).any():
                print("Found NaN/Inf in input features!")
                batch.x = torch.nan_to_num(batch.x, nan=0.0, posinf=0.0, neginf=0.0)
            out = model(x, edge_index)

            # Compute loss
            # train_logits = out[batch.train_mask]
            # train_labels = batch.y[batch.train_mask]
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}") #type: ignore

    torch.save(model.state_dict(), "trojan_gnn_model.pth")
    print("Model saved")


    # Testing
    # model = TrojanGNN(in_dim, hidden_dim, num_classes)
    # model.load_state_dict(torch.load("trojan_gn_model.pth"))

    model.eval()
    correct = 0
    total = 0

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    misclassified_graphs = []

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            x = batch.x
            edge_index = batch.edge_index

            print(f"Testing with {batch.design_name}.")

            # x = torch.nan_to_num(x, nan=normalize_score(10000, "co"), posinf=0.0, neginf=0.0)

            out = model(x, edge_index)
            preds = out.argmax(dim=1)
            labels = batch.y
            # correct += (preds[batch.test_mask] == batch.y[batch.test_mask]).sum().item()

            for i in range(len(labels)):
                pred = preds[i].item()
                true = labels[i].item()

                if pred == true:
                    correct += 1
                    if pred == 1:
                        TP += 1
                    else:
                        TN += 1
                else:
                    misclassified_graphs.append(idx)
                    if pred == 1:
                        FP += 1
                    else:
                        FN += 1

                total += 1

    # Compute rates
    accuracy = correct / total if total > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print results
    print()
    print(f"Test Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    print(f"False Positive Rate: {FPR*100:.2f}%")
    print(f"False Negative Rate: {FNR*100:.2f}%")
    print(f"True Positive Rate:  {TPR*100:6.2f}%")
    print(f"True Negative Rate:  {TNR*100:6.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1 Score: {f1_score*100:.2f}%")
    print(f"Total Misclassified nodes: {len(misclassified_graphs)}")
    # print(f"Indices of misclassified graphs: {misclassified_graphs}")

class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor|None = None, gamma: float = 2.0, reduction='mean'):
        super().__init__()
        self.weight    = weight
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # compute per‑node CE loss without reduction
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='mean')
        pt = torch.exp(-ce)                # pt = model’s prob of the true class
        fl = ((1-pt)**self.gamma) * ce     # focal scaling
        if self.reduction=='mean':
            return fl.mean()
        if self.reduction=='sum':
            return fl.sum()
        return fl
        
class TrojanGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(TrojanGNN, self).__init__()

        # Layer 1: in_dim → hidden_dim
        self.conv1 = GCNConv(in_dim,    hidden_dim)
        self.bn1   = BatchNorm(hidden_dim)

        # Layer 2: hidden_dim → hidden_dim
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2   = BatchNorm(hidden_dim)

        # Layer 3: hidden_dim → hidden_dim
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3   = BatchNorm(hidden_dim)

        # Layer 4 (output): hidden_dim → num_classes
        self.conv4 = GCNConv(hidden_dim, num_classes)

        # Dropout probability
        self.dropout_p = 0.01

    def forward(self, x, edge_index):
        # 1st conv block
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # 2nd conv block
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # 3rd conv block
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # Final conv → class logits
        x = self.conv4(x, edge_index)

        return x
# class TrojanGNN(torch.nn.Module):
#     def __init__(self, in_dim, hidden_dim, num_classes):
#         super(TrojanGNN, self).__init__()
#         # First graph convolution layer
#         self.conv1 = GCNConv(in_dim, hidden_dim)
#         # Batch normalization for hidden dimension
#         self.bn1 = BatchNorm(hidden_dim)
#         # Second graph convolution layer for output
#         self.conv2 = GCNConv(hidden_dim, num_classes)
#         # Dropout probability
#         self.dropout_p = 0.001

#     def forward(self, x, edge_index):
#         # First convolution
#         x1 = self.conv1(x, edge_index)
#         # Apply batch normalization
#         x2 = self.bn1(x1)
#         # Apply ReLU activation
#         x3 = F.relu(x2)
#         # Apply dropout
#         x4 = F.dropout(x3, p=self.dropout_p, training=self.training)
#         # Second convolution
#         x5 = self.conv2(x4, edge_index)
#         return x5

# class TrojanSAGENet(torch.nn.Module):
#     def __init__(self, in_dim, hidden_dim, num_classes):
#         super().__init__()
#         # First SAGE conv
#         self.conv1 = SAGEConv(in_dim, hidden_dim)
#         self.bn1   = BatchNorm(hidden_dim)
#         # Second SAGE conv
#         self.conv2 = SAGEConv(hidden_dim, hidden_dim)
#         self.bn2   = BatchNorm(hidden_dim)
#         # Final linear classification head
#         self.lin   = torch.nn.Linear(hidden_dim, num_classes)
#         self.dropout_p = 0.1

#     def forward(self, x, edge_index):
#         # Layer 1
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout_p, training=self.training)

#         # Layer 2
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout_p, training=self.training)

#         # Classification
#         out = self.lin(x)
#         return out

if __name__ == '__main__':
    # Run this part to save datasets as a file #########################

    compute_scoap_stats()
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