"""
Neural network models for ADMET prediction.

Implements:
- D-MPNN/Chemprop-style message passing
- GIN (Graph Isomorphism Network)

Design principles:
- Minimal, stable implementations
- Standard configurations from literature
- Resume-friendly checkpoint saving
- TDC official split compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

from .smiles_compat import model_input_smiles


class MolecularGraphFeaturizer:
    """Convert SMILES to PyTorch Geometric graph data."""

    def __init__(self):
        # Atom features: one-hot atomic number (up to 100) + degree + formal charge + hybridization
        self.atom_features_dim = 100 + 6 + 5 + 6  # 117 total

    def smiles_to_graph(self, smiles):
        """Convert SMILES to PyG Data object."""
        # Keep the official SMILES as the sample identity in prediction files,
        # but repair two audited legacy N-oxide encodings for modern RDKit.
        mol = Chem.MolFromSmiles(model_input_smiles(smiles))
        if mol is None:
            return None

        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            feat = self._atom_features(atom)
            atom_features.append(feat)

        x = torch.tensor(atom_features, dtype=torch.float32)

        # Edge indices (undirected graph)
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])  # undirected

        if len(edge_indices) == 0:
            # Single atom molecule - add self-loop
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)

    def _atom_features(self, atom):
        """Extract atom features."""
        features = []

        # Atomic number (one-hot up to 100)
        atomic_num = atom.GetAtomicNum()
        one_hot = [0] * 100
        if atomic_num < 100:
            one_hot[atomic_num] = 1
        features.extend(one_hot)

        # Degree (one-hot up to 6)
        degree = atom.GetDegree()
        degree_one_hot = [0] * 6
        if degree < 6:
            degree_one_hot[degree] = 1
        features.extend(degree_one_hot)

        # Formal charge (one-hot from -2 to +2)
        charge = atom.GetFormalCharge()
        charge_one_hot = [0] * 5
        charge_idx = charge + 2  # map -2..+2 to 0..4
        if 0 <= charge_idx < 5:
            charge_one_hot[charge_idx] = 1
        features.extend(charge_one_hot)

        # Hybridization (one-hot: SP, SP2, SP3, SP3D, SP3D2, other)
        hybrid = str(atom.GetHybridization())
        hybrid_one_hot = [0] * 6
        hybrid_map = {'SP': 0, 'SP2': 1, 'SP3': 2, 'SP3D': 3, 'SP3D2': 4}
        if hybrid in hybrid_map:
            hybrid_one_hot[hybrid_map[hybrid]] = 1
        else:
            hybrid_one_hot[5] = 1  # other
        features.extend(hybrid_one_hot)

        return features


class GINModel(nn.Module):
    """
    Graph Isomorphism Network (Xu et al., 2019).

    Standard configuration:
    - 5 layers
    - 256 hidden dim
    - sum aggregation
    - dropout 0.2
    """

    def __init__(self, input_dim=117, hidden_dim=256, num_layers=5, dropout=0.2, task_type='classification'):
        super().__init__()
        self.task_type = task_type
        self.num_layers = num_layers
        self.dropout = dropout

        # GIN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp, train_eps=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Prediction head
        if task_type == 'classification':
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)  # binary classification
            )
        else:  # regression
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GIN layers with sum aggregation
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = global_add_pool(x, batch)

        # Prediction
        out = self.predictor(x)
        return out.squeeze(-1)


class DMPNNModel(nn.Module):
    """
    Simplified D-MPNN/Chemprop-style model.

    Standard configuration:
    - 3 message passing steps
    - 300 hidden dim
    - edge-based message passing
    - dropout 0.1
    """

    def __init__(self, atom_dim=117, hidden_dim=300, depth=3, dropout=0.1, task_type='classification'):
        super().__init__()
        self.task_type = task_type
        self.depth = depth
        self.dropout = dropout

        # Message passing layers
        self.W_i = nn.Linear(atom_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(atom_dim + hidden_dim, hidden_dim)

        # Prediction head
        if task_type == 'classification':
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        else:  # regression
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Initialize messages
        h = torch.zeros(x.size(0), self.W_i.out_features, device=x.device)

        # Message passing
        for _ in range(self.depth):
            # Collect messages from neighbors
            messages = []
            for node_idx in range(x.size(0)):
                neighbors = (edge_index[0] == node_idx).nonzero(as_tuple=True)[0]
                if len(neighbors) > 0:
                    neighbor_nodes = edge_index[1][neighbors]
                    neighbor_messages = h[neighbor_nodes]
                    aggregated = torch.sum(neighbor_messages, dim=0)
                else:
                    aggregated = torch.zeros(h.size(1), device=x.device)
                messages.append(aggregated)

            messages = torch.stack(messages)

            # Update hidden states
            h_input = self.W_i(x) + self.W_h(messages)
            h = F.relu(h_input)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Readout: concatenate atom features with final hidden states
        node_repr = torch.cat([x, h], dim=1)
        node_repr = self.W_o(node_repr)

        # Global pooling
        mol_repr = global_mean_pool(node_repr, batch)

        # Prediction
        out = self.predictor(mol_repr)
        return out.squeeze(-1)


def collate_graphs(batch):
    """Collate function for DataLoader."""
    # Filter out None values (invalid SMILES)
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return Batch.from_data_list(batch)
