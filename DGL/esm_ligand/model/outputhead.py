import torch
import torch.nn as nn
from unimol_tools.models.transformersv2 import TransformerEncoderWithPairV2
from typing import Tuple

class DenseMLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_hid_layers = 2, activation = nn.ReLU, mlp_dropout = 0.0):
        super(DenseMLP, self).__init__()
        self.activation = activation
        list_of_layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(num_hid_layers):
            list_of_layers.append(nn.Dropout(mlp_dropout))
            list_of_layers.append(self.activation())
            list_of_layers.append(nn.Linear(hid_dim, hid_dim))
        list_of_layers.append(nn.Dropout(mlp_dropout))
        list_of_layers.append(self.activation())
        list_of_layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.Sequential(*list_of_layers)

    def forward(self, x):
        return self.layers(x)


class OutputHead(nn.Module):
    def __init__(self):
        super(OutputHead, self).__init__()
        

    def forward(self, atom_embeddings):
        raise NotImplementedError
    
class MLPOutputHead(nn.Module):
    def __init__(self, coords_head_args: dict, class_head_args: dict, pair_head_args: dict = None):
        super(MLPOutputHead, self).__init__()
        self.coord_head = DenseMLP(**coords_head_args) if coords_head_args else None
        self.class_head = DenseMLP(**class_head_args)
        self.pair_head = DenseMLP(**pair_head_args) if pair_head_args else None

    def forward(self, atom_embeddings, pair_embeddings):
        pred_coord = None
        if self.coord_head:
            pred_coord = self.coord_head(atom_embeddings)
        pred_atom_type = self.class_head(atom_embeddings)
        if self.pair_head:
            pred_pair = self.pair_head(pair_embeddings)
        else:
            pred_pair = None
        return pred_coord, pred_atom_type, pred_pair
