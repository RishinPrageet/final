# data_loader.py
import numpy as np
import torch
import mne
from torch.utils.data import Dataset
from torch_geometric.data import Data

# ---------------------------
# EEG loading (EEGLAB .set)
# ---------------------------
def load_eeg_file(path):
    raw = mne.io.read_raw_eeglab(
        path,
        preload=True,
        verbose=False
    )

    # sanity checks from metadata
    assert raw.info["sfreq"] == 500, "Unexpected sampling frequency"
    assert raw.info["nchan"] == 19, "Unexpected channel count"

    raw.filter(1., 45., fir_design="firwin")
    return raw.get_data()  # (19, T)


# ---------------------------
# Graph utilities
# ---------------------------
def compute_corr_adj(eeg_window):
    corr = np.corrcoef(eeg_window)
    corr = np.nan_to_num(corr)
    return np.abs(corr)

def sparsify_adj(adj, k=4):  # k adapted for 19 channels
    A = np.zeros_like(adj)
    for i in range(adj.shape[0]):
        idx = np.argsort(adj[i])[-k:]
        A[i, idx] = adj[i, idx]
    return A

def adj_to_edge_index(adj):
    src, dst = np.where(adj > 0)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(adj[src, dst], dtype=torch.float)
    return edge_index, edge_attr

def node_features(win):
    mean = win.mean(axis=1)
    std = win.std(axis=1)
    return np.stack([mean, std], axis=1)  # (19, 2)

# ---------------------------
# Dataset
# ---------------------------
class EEGGraphDataset(Dataset):
    def __init__(
        self,
        files,
        labels,
        client_id,
        window_seconds=4,
        sfreq=500,
        k=4,
    ):
        self.samples = []
        self.client_id = client_id
        self.window_samples = window_seconds * sfreq
        self.k = k
        self._prepare(files, labels)

    def _prepare(self, files, labels):
        for f, y in zip(files, labels):
            data = load_eeg_file(f)  # (19, T)
            T = data.shape[1]
            step = self.window_samples // 2

            for start in range(0, T - self.window_samples + 1, step):
                win = data[:, start:start + self.window_samples]

                # channel-wise normalization
                win = (win - win.mean(axis=1, keepdims=True)) / \
                      (win.std(axis=1, keepdims=True) + 1e-8)

                adj = compute_corr_adj(win)
                adj = sparsify_adj(adj, self.k)
                edge_index, edge_attr = adj_to_edge_index(adj)

                x = node_features(win)

                self.samples.append(
                    Data(
                        x=torch.tensor(x, dtype=torch.float),
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=torch.tensor(y, dtype=torch.long),
                        domain=torch.tensor(self.client_id, dtype=torch.long),
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
