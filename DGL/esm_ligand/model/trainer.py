import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import time

from model.pretrain_module import Pretrain_module
from torch.utils.data.dataloader import DataLoader
from model.utils import calculate_rmsd_batch

class Trainer():
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.log = defaultdict(list)
        
    
    def _iter_epoch(self, dataloader: DataLoader, epoch: int, train = True):
        raise NotImplementedError
    

class PreTrainer_V1(Trainer):
    def __init__(self, model: Pretrain_module, optimizer: torch.optim.Optimizer, device , use_classification_loss, commitment_beta, classification_loss_weight, dist_loss_cap, use_clip):
        """
        Initializes the PreTrainer_V1 class.

        Args:
            model (Pretrain_module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            device: The device (CPU or GPU) to run the model on.
            use_classification_loss (bool): Whether to use classification loss during training.
            commitment_beta (float): Beta parameter for the commitment loss.
            classification_loss_weight (float): Weight for the classification loss.
            dist_loss_cap (float): Maximum cap for the distance loss.
            use_clip (bool): Whether to use gradient clipping during training.
        """
        super(PreTrainer_V1, self).__init__(model, optimizer)
        self.input_mask_probs = [0.0, 0.0, 0.0, 0.0]
        self.device = device
        self.use_classification_loss = use_classification_loss
        self.commitment_beta = commitment_beta
        self.classification_loss_weight = classification_loss_weight
        self.dist_loss_cap = dist_loss_cap
        self.use_clip = use_clip
        
	
    def _iter_epoch(self, dataloader: DataLoader, epoch: int, train: bool):
        if train:
            self.model.train()
            desc = f"Training epoch {epoch}"
        else:
            self.model.eval()
            desc = "Iterating over validation data"
            torch.no_grad()

        # Accumulators: track total value
        # These track the total sum over all batches. 
		# At the end of the epoch, we divide by the number of valid atoms/pairs to get the mean.
        accum_num_atom = 0
        accum_num_pair = 0
        accum_class_loss = 0
        accum_dist_loss = 0
        accum_rmsd = 0
        accum_perplexity = []
        
        for batch in tqdm(dataloader, desc=desc):
            #move tensors to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            #forward pass
            pred_coord, pred_atom_type, pred_pair, dictionary_loss, commitment_loss, perplexity, heavy_pair_mask, heavy_atom_mask = self.model(batch, self.input_mask_probs)
            
            rmsds, pairwise_dist_loss = self._distance_based_losses(batch, pred_coord, pred_pair, heavy_atom_mask, heavy_pair_mask)
            accum_rmsd += torch.sum(rmsds).item()
            accum_dist_loss += torch.sum(pairwise_dist_loss).item()
            #normalized loss for this batch.
            pairwise_dist_loss = torch.sum(pairwise_dist_loss) / torch.sum(heavy_pair_mask)
            
            #calculate atom type classification loss
            atom_type_classification_loss = self._atom_type_class_loss(batch, pred_atom_type, heavy_atom_mask)
            accum_class_loss += torch.sum(atom_type_classification_loss).item()
            atom_type_classification_loss = torch.sum(atom_type_classification_loss) / torch.sum(heavy_atom_mask)
            
            #total loss
            total_loss = torch.mean(rmsds)+ pairwise_dist_loss + commitment_loss * self.commitment_beta
            if dictionary_loss:
                total_loss += dictionary_loss
            if self.use_classification_loss:
                total_loss += atom_type_classification_loss * self.classification_loss_weight
                
            accum_num_atom += torch.sum(heavy_atom_mask).item()
            accum_num_pair += torch.sum(heavy_pair_mask ** 2).item()
            accum_perplexity.append(perplexity)
            
            if train:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        
        if not torch.is_grad_enabled():
            torch.enable_grad()
        
        avg_class_loss = accum_class_loss / accum_num_atom
        avg_dist_loss = accum_dist_loss / accum_num_pair
        mean_perplexity = np.mean(accum_perplexity)
        mean_rmsd = accum_rmsd / len(dataloader.dataset)

        return avg_class_loss, avg_dist_loss, mean_perplexity, mean_rmsd
        
    def train(self, epochs: int, train_loader: DataLoader, val_loader: DataLoader, write_log = True, log_name = None):
        for epoch in range(1, epochs + 1):
            start = time.time()
            class_loss, dist_loss, perplexity, rmsd = self._iter_epoch(train_loader, epoch, True)
            self._log(class_loss, dist_loss, perplexity, rmsd, train = True)
            
            val_class_loss, val_dist_loss, val_perplexity, val_rmsd = self.eval(val_loader)
            self._log(val_class_loss, val_dist_loss, val_perplexity, val_rmsd, train = False)
            
            if write_log:
                end = time.time()
                log_name = "log.txt" if not log_name else log_name
                self._write_log_to_file(log_name, end - start, epoch)

    def eval(self, dataloader: DataLoader):
        return self._iter_epoch(dataloader, 0, False)

    def _log(self, class_loss, dist_loss, mean_perplexity, rmsd, train = True):
        if train:
            self.log['train_class_loss'].append(class_loss)
            self.log['train_dist_loss'].append(dist_loss)
            self.log['train_perplexity'].append(mean_perplexity)
            self.log['train_rmsd'].append(rmsd)
        else:
            self.log['val_class_loss'].append(class_loss)
            self.log['val_dist_loss'].append(dist_loss)
            self.log['val_perplexity'].append(mean_perplexity)
            self.log['val_rmsd'].append(rmsd)
            
    def _write_log_to_file(self, log_file_name: str, time, epoch: int):
        with open(log_file_name, 'a') as f:
            f.write(f"Epoch {epoch} finished.\n")
            f.write(f"Time cost: {time}\n")
            f.write(f"train_class_loss: {self.log['train_class_loss'][-1]}, train_dist_loss: {self.log['train_dist_loss'][-1]}, train_perplexity: {self.log['train_perplexity'][-1]}, rmsd: {self.log['train_rmsd'][-1]}\n")
            f.write(f"val_class_loss: {self.log['val_class_loss'][-1]}, val_dist_loss: {self.log['val_dist_loss'][-1]}, val_perplexity: {self.log['val_perplexity'][-1]}, rmsd: {self.log['val_rmsd'][-1]}\n")

    def _distance_based_losses(self, batch_data, pred_coord, pair_pred, atom_mask, pair_mask):
        pairwise_dist_loss_fn = torch.nn.MSELoss(reduction = "none")

        rmsds = calculate_rmsd_batch(pred_coord, batch_data['conformation'], atom_mask)

        if pair_pred is None:
            pred_dist_matrix = torch.cdist(pred_coord, pred_coord, p = 2)
        else:
            pred_dist_matrix = pair_pred.squeeze()
        pairwise_dist_loss = pairwise_dist_loss_fn(pred_dist_matrix, batch_data['dist']) * pair_mask
        if self.use_clip:
            pairwise_dist_loss = torch.clip(pairwise_dist_loss, max = self.dist_loss_cap)
        return rmsds, pairwise_dist_loss
    
    def _atom_type_class_loss(self, batch_data, pred_atom_type, atom_mask):
        atom_type_classification_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        atom_type_classification_loss = atom_type_classification_loss_fn(torch.permute(pred_atom_type,  (0,2,1)), batch_data['atomic_numbers']) * atom_mask
        return atom_type_classification_loss
    
    def set_device(self, device):
        self.device = device
	