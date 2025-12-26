# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm

class STGNNFeatureExtractor(nn.Module):
    def __init__(self, in_feats=2, hidden=64, out_feats=128, heads=4, dropout=0.2):
        super().__init__()

        self.gnn1 = GATConv(in_feats, hidden, heads=heads, concat=True, dropout=dropout)
        self.bn1 = BatchNorm(hidden * heads)

        self.gnn2 = GATConv(hidden * heads, out_feats, heads=1, concat=False, dropout=dropout)
        self.bn2 = BatchNorm(out_feats)

        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gnn1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gnn2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)

        return global_mean_pool(x, batch)

class EEGGraphClassifier(nn.Module):
    def __init__(self, node_feat_dim=2, embed_dim=128, num_classes=2):
        super().__init__()

        self.encoder = STGNNFeatureExtractor(
            in_feats=node_feat_dim,
            out_feats=embed_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, data):
        feats = self.encoder(data)
        return self.classifier(feats)

    def extract_features(self, data):
        return self.encoder(data)
