# train_local.py
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from data_loader import EEGGraphDataset
from model import EEGGraphClassifier
from label_utils import load_file_label_pairs

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# Paths (BIDS root)
# -------------------------------------------------
root_dir = r"C:\Users\kira7\Downloads\ds004504-main\ds004504-main"
participants_tsv = rf"{root_dir}\participants.tsv"

# -------------------------------------------------
# Load files & labels safely
# -------------------------------------------------
files, labels = load_file_label_pairs(root_dir, participants_tsv)

print("Label distribution:", {0: labels.count(0), 1: labels.count(1)})
assert len(set(labels)) == 2, "ERROR: Need both Alzheimer and Control subjects"

# -------------------------------------------------
# Simple subject-wise split (TEMP)
# -------------------------------------------------
split = int(0.8 * len(files))
train_files, val_files = files[:split], files[split:]
train_labels, val_labels = labels[:split], labels[split:]

# -------------------------------------------------
# Dataset & loaders
# -------------------------------------------------
train_ds = EEGGraphDataset(train_files, train_labels, client_id=0)
val_ds   = EEGGraphDataset(val_files, val_labels, client_id=0)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)

# -------------------------------------------------
# Model
# -------------------------------------------------
model = EEGGraphClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------------------------------------
# Training loop
# -------------------------------------------------
for epoch in range(10):
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        logits = model(data)
        loss = F.cross_entropy(logits, data.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    train_loss = total_loss / len(train_loader.dataset)

    model.eval()
    correct = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            preds = model(data).argmax(dim=1)
            correct += (preds == data.y).sum().item()

    val_acc = correct / len(val_loader.dataset)

    print(f"Epoch {epoch}: loss={train_loss:.4f}, val_acc={val_acc:.4f}")
