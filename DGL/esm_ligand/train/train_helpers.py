#from @andrew0901

import torch
import numpy as np
import random
import os

def set_seed(seed):
    """Sets the random seed for reproducibility across PyTorch, NumPy, and Python's random module."""
    torch.manual_seed(seed)  # Seed for CPU and CUDA operations
    torch.cuda.manual_seed(seed)  # Seed for specific CUDA operations (if using multiple GPUs)
    torch.cuda.manual_seed_all(seed) # Seed for all GPUs
    np.random.seed(seed)  # Seed for NumPy operations
    random.seed(seed)  # Seed for Python's built-in random module
    os.environ['PYTHONHASHSEED'] = str(seed) # Set a fixed value for the hash seed