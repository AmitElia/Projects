import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

from train.train_args import *
from train.train_helpers import set_seed
from data.dataset import *

from model.featurizer import Featurizer_V1
from model.encoder import UniMolEncoder
from model.quantizer import VectorQuantizer
from model.decoder import VanillaTransformerDecoder
from model.outputhead import MLPOutputHead
from model.trainer import PreTrainer_V1


set_seed(310800)

from model.pretrain_module import Pretrain_module

dataset = Spice2Dataset(os.path.join(os.path.dirname(__file__), "tiny.npz"))
train_set, val_set = random_split(dataset, [0.85, 0.15])
train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate_molecular_data, shuffle=True)
val_loader = DataLoader(val_set, batch_size=VAL_BATCH_SIZE, collate_fn=collate_molecular_data, shuffle=False)

featurizer = Featurizer_V1(**featurizer_args)
encoder = UniMolEncoder(encoder_args)
quantizer = VectorQuantizer(**quantizer_args)
decoder = VanillaTransformerDecoder(decoder_layer_args,decoder_args)
outputhead = MLPOutputHead(coord_head_args, type_head_args)

pretrain_module = Pretrain_module(featurizer, encoder, quantizer, decoder, outputhead)
optimizer = Adam(pretrain_module.parameters(), lr=LR)
trainer = PreTrainer_V1(pretrain_module, optimizer, **trainer_args)

print("Initialization finished.")
trainer.model.cuda()
trainer.set_device(torch.device("cuda"))
trainer.train(10, train_loader, val_loader, write_log=True)