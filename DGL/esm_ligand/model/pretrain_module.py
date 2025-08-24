import torch
import torch.nn as nn

from featurizer import Featurizer
from encoder import Encoder
from quantizer import Quantizer
from decoder import Decoder
from outputhead import OutputHead
from typing import List, Dict

class Pretrain_module(nn.Module):
    def __init__(self, featurizer: Featurizer, encoder: Encoder, quantizer: Quantizer, 
                 decoder: Decoder, output_head: OutputHead):
        super(Pretrain_module, self).__init__()
        self.featurizer = featurizer
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.output_head = output_head
        self.device = torch.device("cpu")

    def forward(self, batch_data, mask_probs:List[float] = [0.0, 0.0, 0.0, 0.0]):
        '''
        Forward pass for pretraining module.
        Args:
         - batch_data: Dictionary containing input data for featurization.
         - mask_probs: List of probabilities for different types of masking.
        Returns:
        '''
        #mask of valid atoms
        atom_mask = batch_data['atom_mask']
        pair_mask = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)
        
        #remove hydrogen or dummy atoms. used to focus loss and quantization on heavy atoms only
        heavy_atom_mask = atom_mask * (batch_data['atomic_number'] != 0).int()
        heavy_pair_mask = heavy_atom_mask.unsqueeze(-1) * heavy_atom_mask.unsqueeze(-2)
        
        #randomly sample masks based on the provided probabilities
        masks = self.featurizer.sample_masks(atom_mask, *mask_probs)
        #featurize the input data
        atom_features, pairwise_features, topological_features = self.featurizer(batch_data, masks)

        #encode the features
        #TODO
        atom_embeddings, pairwise_embeddings, coords = self.encoder(atom_features, pairwise_features, batch_data['conformation'], atom_mask, pair_mask)
        
        #quantize the atom embeddings
        #TODO,  ?Why heavy_atom_mask?
        quantized_atom_features, dictionary_loss, commitment_loss, encoding_indices, perplexity = self.quantizer(atom_embeddings, heavy_atom_mask)
        
        padding_mask = (1 - heavy_atom_mask).bool() #for Vanilla Transformer decoder, padding mask is 1 for padding atoms, 0 for real atoms
        atom_out_embedding, pair_out_embedding, coords_out = self.decoder(quantized_atom_features, batch_data['topological_dist'].unsqueeze(-1),
                                                                           coords, heavy_atom_mask, heavy_pair_mask, None, padding_mask)
        pred_coord, pred_atom_type, pred_pair = self.output_head(atom_out_embedding, pair_out_embedding)
        
        return pred_coord, pred_atom_type, pred_pair, dictionary_loss, commitment_loss, perplexity, heavy_pair_mask, heavy_atom_mask
        
    def set_device(self, device):
        self.device = device
		
        
        
