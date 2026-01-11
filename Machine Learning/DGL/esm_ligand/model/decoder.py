import torch
import torch.nn as nn
from unimol_tools.models.transformersv2 import TransformerEncoderWithPairV2
from typing import Tuple

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, atom_feats, pair_feats, coords, atom_mask, pair_mask, attn_mask, padding_mask) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        #decode information for pre-training tasks, atom_embeddings of shape (B, L, E)
        raise NotImplementedError
    
class VanillaTransformerDecoder(Decoder):
    #Wrapper class for decoder using vanilla transformers from Attention is All You Need
    def __init__(self, decoder_layer_args: dict, decoder_args: dict):
        super(VanillaTransformerDecoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(**decoder_layer_args)
        self.decoder = nn.TransformerEncoder(encoder_layer, **decoder_args)
    
    def forward(self, atom_feats, pair_feats, coords, atom_mask, pair_mask, attn_mask, padding_mask):
        #note for mask:  If a BoolTensor is provided, the positions with the value of True will be ignored while the position with the value of False will be unchanged.
        out = self.decoder(atom_feats, src_key_padding_mask = padding_mask)
        return out, None, None
    
class UniMolDecoder(Decoder):
    #Wrapper class for using attention with pair architecture as decoder
    def __init__(self, encoder_args: dict):
        super(UniMolDecoder, self).__init__()
        self.decoder = TransformerEncoderWithPairV2(**encoder_args)
        self.pair_encoding = nn.Linear(1, encoder_args['pair_dim'])

    def forward(self, atom_feats: torch.Tensor, pair_feats: torch.Tensor, coords,
                atom_mask: torch.Tensor, pair_mask: torch.Tensor, attn_mask: torch.Tensor = None, padding_mask = None) -> Tuple[torch.Tensor, torch.Tensor]:
        #for shapes of each input, see definition in forward method of Encoder class
        pair_feats = self.pair_encoding(pair_feats.float())
        atom_embedding, pair_embedding = self.decoder(atom_feats, pair_feats, atom_mask, pair_mask, attn_mask)
        return atom_embedding, pair_embedding, coords