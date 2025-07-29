import torch
import torch.nn as nn
from typing import Dict, Tuple
import model_constants as C

class Featurizer(nn.Module):
    def __init__(self):
        super(Featurizer, self).__init__()
    
    def forward(self, data: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
    def sample_masks(self, atom_mask: torch.Tensor, atomic_number_mask_prob: float = 0, 
                     atom_coord_mask_prob: float = 0, atom_complete_mask_prob: float = 0, charge_mask_prob: float = 0) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    

class Featurizer_V1(Featurizer):
    def __init__(self, atom_embedding_dim, pair_dim, num_atom_type):
        '''
        Featurizer_V1 is the first version of the featurizer. 
        It encodes both atom-level and pairwise features using embedding layers and optional masking.
		The input data is expected to be a dictionary containing:
		- 'dist': Euclidean distances between atoms (tensor of shape [num_pairs])
		- 'topological_dist': Topological distances (tensor of shape [num_pairs])
		- 'atomic_number': Atomic numbers of atoms (tensor of shape [num_atoms])
		- 'charge': Charges of atoms (tensor of shape [num_atoms])
		- 'chirality': Chirality information of atoms (tensor of shape [num_atoms])
        '''
        super(Featurizer_V1, self).__init__()
        assert pair_dim % 2 == 0, "pair_dim has to be divisible by 2"
        
		# Converts bucketized Euclidean distances into learnable embeddings.
        self.euclidean_dist_encoding = nn.Embedding(C.NUM_DISTANCE_BUCKET, pair_dim)
        # Converts continuous distances to discrete bins for embedding lookup.
        self.bucket_dist = lambda x: torch.bucketize(x, torch.arange(0, C.DISTANCE_UPPER_LIMIT, step=C.DISTANCE_BIN_SIZE).to(x.device))

		# Topological distance (graph-based, integer-valued) is also bucketed and embedded.
        self.topological_dist_encoding = nn.Embedding(C.TOPO_DIST_UPPER_LIMIT + 1, pair_dim) 
        self.bucket_topo_dist = lambda x: torch.bucketize(x, torch.arange(0, C.TOPO_DIST_UPPER_LIMIT).to(x.device))

		# Atom features (type, charge, chirality) are embedded separately and then summed.
        self.atom_type_encoding = nn.Embedding(num_atom_type, atom_embedding_dim)  # Atomic number
        self.charge_encoding = nn.Embedding(C.NUM_CHARGE_BUCKET, atom_embedding_dim)  # Charge
        self.bucket_charge = lambda x: torch.bucketize(x, torch.arange(C.CHARGE_LOWER_LIMIT, C.CHARGE_UPPER_LIMIT, step=C.CHARGE_BIN_SIZE).to(x.device))
        self.chirality_encoding = nn.Embedding(3, atom_embedding_dim)  # R, S and no chirality

	
    def forward(self, data: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
    	Input data has distances, charges, atom types, etc.
		Optional masks allow dropout-style masking for pretraining.
        
        returns:
         -atom_embeddings
         -pair
         -encoded_topo
		'''
        # A tensor of shape [N, N] with pairwise 3D distances (in Å)
        euclidean_dist = data['dist']
        # A tensor of shape [N, N] with bond graph distances
        topological_dist = data['topological_dist']
        
        device = euclidean_dist.device
        
		# loading masks if provided, otherwise creating default tensors
        coord_mask = masks['coord_mask'] if masks else torch.tensor(1).to(device)
        atomic_number_mask = masks['atomic_number_mask'] if masks else torch.tensor(1).to(device)
        atom_complete_mask = masks['complete_mask'] if masks else torch.tensor(1).to(device)
        atom_complete_mask_for_pairs = masks['complete_mask_for_pairs'] if masks else torch.tensor(1).to(device)
        charge_mask = masks['charge_mask'] if masks else torch.tensor(1).to(device)
        
		# bucket_dist: Converts continuous Euclidean distances to discrete bins.
		# euclidean_dist_encoding/ topological_dist_encoding: Embeds bucketed values → shape [N, N, pair_dim].
		# Masking: Multiplied by coord_mask.unsqueeze(-1) → zeroes out masked entries.
        euclidean_dist_encoded = self.euclidean_dist_encoding(self.bucket_dist(euclidean_dist)) * coord_mask.unsqueeze(-1)
        topological_dist_encoded = self.topological_dist_encoding(self.bucket_topo_dist(topological_dist))
        

        pairwise_features = euclidean_dist_encoded + topological_dist_encoded
        pairwise_features *= atom_complete_mask_for_pairs.unsqueeze(-1)
        
		# [N, atom_embedding_dim], multiplied by atom_complete_mask to zero out masked atoms.
        atom_type_encoded = self.atom_type_encoding(data['atomic_numbers']) * atomic_number_mask.unsqueeze(-1)
        charge_encoded = self.charge_encoding(self.bucket_charge(data['charges'])) * charge_mask.unsqueeze(-1)
        chirality_encoded = self.chirality_encoding(data['chiral_mask'])
        
        atom_embeddings = atom_type_encoded + charge_encoded + chirality_encoded
        #apply complete mask to atom embeddings
        atom_embeddings *= atom_complete_mask.unsqueeze(-1)
        

        return atom_embeddings, pairwise_features, topological_dist_encoded
        

    def sample_masks(self, atom_mask: torch.Tensor, atomic_number_mask_prob: float = 0,
                     atom_coord_mask_prob: float = 0, atom_complete_mask_prob: float = 0, charge_mask_prob: float = 0) -> Dict[str, torch.Tensor]:
        '''
        This function simulates random masking of features for pretraining.
            
        atom_mask: Binary tensor of shape [N], where 1 = real atom, 0 = padding.
        Probabilities control how often features are masked.
        Returns a dictionary of masks used in the forward pass.
        '''
        device = atom_mask.device
        # unsqueeze(-1) and unsqueeze(-2) expand dimensions to [N, 1] and [1, N]. Outer product → shape [N, N]
        # pair_mask[i, j] = 1 only if both atoms i and j exist
        pair_mask = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)
        
        # For each atom: If value > atom_complete_mask_prob → keep (1). Else → mask (0)
        # A binary mask (shape [N]) indicating whether to keep or drop the entire atom.
        complete_mask = (torch.rand(atom_mask.shape).to(device) > atom_complete_mask_prob).int()
        # Ensures that if atom i is masked, all pair[i][*] and pair[*][i] are also masked. Only pairs of non-masked atoms are retained.
        complete_mask_for_pairs = complete_mask.unsqueeze(-1) * complete_mask.unsqueeze(-2)
        
		# only masks coordinates
        atom_coord_mask = (torch.rand(atom_mask.shape).to(device) > atom_coord_mask_prob).int()
        # if either atom is masked, their pairwise geometry is also masked.
        coord_mask = atom_coord_mask.unsqueeze(-1) * atom_coord_mask.unsqueeze(-2)

        # only masks atomic numbers
        atomic_number_mask = (torch.rand(atom_mask.shape).to(device) > atomic_number_mask_prob).int()
        # ensure padding atoms remain masked
        atomic_number_mask *= atom_mask
        
		# only masks charges
        charge_mask = (torch.rand(atom_mask.shape).to(device) > charge_mask_prob).int()
        # ensure padding atoms remain masked
        charge_mask *= atom_mask
        
        masks = {
            'complete_mask': complete_mask,
            'complete_mask_for_pairs': complete_mask_for_pairs,
            'atomic_number_mask': atomic_number_mask,
            'coord_mask': coord_mask,
            'charge_mask': charge_mask
        }
        return masks