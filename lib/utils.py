import os
import torch
from pytorch_lightning import seed_everything
# import random
import numpy as np
import torch.nn as nn

def set_reproducibility(random_seed=42):
    # random.seed(random_seed)
    # np.random.seed(random_seed)
    # torch.manual_seed(random_seed)
    # torch.cuda.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    seed_everything(random_seed, workers=True) # when workers turned on, it ensures that e.g. data augmentations are not repeated across workers.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model_state(model, file_path=None, validation_id=None):
    """
    save checkpoint (optimizer and model)
    :param validation_id:
    :param model:
    :return:
    """
    if file_path is None:
        return
    file_path = f'{os.path.splitext(file_path)[0]}_{str(validation_id)}.pt'
    print('Saving model: ' + file_path)
    model = model.module if type(model) is torch.nn.DataParallel else model
    torch.save(model.state_dict(), file_path)


def load_model_state(model, state_path):
    # load model onto cpu
    state = torch.load(state_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state)


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def loop_iterable(iterable):
    while True:
        yield from iterable


def make_labels(counts, is_presence=True):
    if is_presence:
        labels = np.ones(counts, dtype=np.float32)
    else:
        labels = np.zeros(counts, dtype=np.float32)
    return labels


def weight_reset(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        m.reset_parameters()