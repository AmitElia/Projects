import torch
LR = 1e-4

ATOM_DIM, PAIR_DIM = 512, 64
NUM_ATOM_TYPE = 16
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 8
featurizer_args = {'atom_embedding_dim': ATOM_DIM, 'pair_dim': PAIR_DIM, 'num_atom_type': NUM_ATOM_TYPE}

encoder_args = {
    "num_encoder_layers": 4,
    "embedding_dim": ATOM_DIM,
    "pair_dim": PAIR_DIM,
    "pair_hidden_dim": 128,
    "num_attention_heads": 8,
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "activation_dropout": 0,
    "activation_fn": "gelu",
    "droppath_prob": 0,
    "pair_dropout": 0.25
    }
quantizer_args = {
    "embedding_dim": ATOM_DIM,
    "num_embeddings": 1024, 
    "use_ema": True,
    "decay": 0.99,
    "epsilon": 1e-03,
    "initialize_limit": 3,
    "with_BN": True
    }
decoder_layer_args = {
    "d_model": ATOM_DIM, 
    "nhead": 8, 
    "dim_feedforward": 1024, 
    "dropout": 0.1, 
    "activation": "relu", 
    "layer_norm_eps": 1e-05, 
    "batch_first": True,
    "norm_first": False, 
    "bias": True, 
    "device": None, 
    "dtype": None
}
decoder_args = {
	"num_layers": 6, 
	"norm": None
}
trainer_args = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_classification_loss": False,
    "commitment_beta": 0.25,
    "classification_loss_weight": 1,
    "dist_loss_cap": 25,
    "use_clip": False
}
coord_head_args = {
    "in_dim": ATOM_DIM, 
    "hid_dim": 512, 
    "out_dim": 3, 
    "num_hid_layers": 2, 
    "activation": torch.nn.GELU, 
    "mlp_dropout": 0.1
}

type_head_args = {
    "in_dim": ATOM_DIM, 
    "hid_dim": 512, 
    "out_dim": NUM_ATOM_TYPE, 
    "num_hid_layers": 2, 
    "activation": torch.nn.GELU, 
    "mlp_dropout": 0.1
}
