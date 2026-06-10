import os
import random

import numpy as np
import torch

DEFAULT_RANDOM_SEED = 17


def seed_basic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def seed_torch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# basic + torch
def seed_everything(seed=DEFAULT_RANDOM_SEED):
    seed_basic(seed)
    seed_torch(seed)
