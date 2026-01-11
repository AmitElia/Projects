###
# encoder.py	
###

import torch
import torch.nn as nn
from typing import Dict, Tuple
from unimol_tools.models.transformersv2 import TransformerEncoderWithPairV2

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()

	def forward(self, atom_feats, pair_feats, coords, atom_mask, pair_mask, attn_mask):
		#encode input features/encodings
        #atom features are of shape (B, L, Ea)
        #pair features are of shape (B, L, L, Ep)
        #coords are of shape (B, L, 3)
        #atom_mask is of shape (B, L)
        #pair_mask and attn_mask are of shape (B, L, L)
        # Since different encoder structure returns different things, 
        # for consistancy, I will make each child class return (scalar feats, pair scalar feats, vector feats)
        # each of shape (B, L, Ea), (B, L, L, Ep), (B, L, 3)
        # If any child class doesn't return pair features or equivariant vector features, corresponding positions will be replaced by None.
        # This will be handled by downstream classes
		raise NotImplementedError
	
class UniMolEncoder(Encoder):
	#wrapper class for UniMol2's encoder layer architecture
	#https://arxiv.org/abs/2406.14969
	def __init__(self, encoder_args: dict):
		super(UniMolEncoder, self).__init__()
		self.encoder = TransformerEncoderWithPairV2(**encoder_args)

	def forward(self, atom_feats: torch.Tensor, pair_feats: torch.Tensor, 
			 coords: torch.Tensor, atom_mask: torch.Tensor, pair_mask: torch.Tensor, attn_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		
		atom_embedding, pair_embedding = self.encoder(atom_feats, pair_feats, atom_mask, pair_mask, attn_mask)
		return atom_embedding, pair_embedding, coords
	


	
