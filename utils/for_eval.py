from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import numpy as np

import torch

from umap import UMAP


def accuracy(model, ds, device):
    truth, embedding = [], []
    model.eval()
    for x, y in ds:
        embedding.append(model.encode(x.to(device)))        
        truth.append(y)
    embedding = np.concatenate(embedding)
    manifold = model.manifold(embedding)
    pred = model.cluster(manifold).argmax(1)
    
    print('reduce the dimension of data for plotting ...')
    embed_plot = UMAP().fit_transform(embedding)

    confusion_m = confusion_matrix(torch.cat(truth).numpy(), pred)
    _, col_idx = linear_sum_assignment(confusion_m, maximize=True)
    acc = np.trace(confusion_m[:, col_idx]) / confusion_m.sum()
    
    return acc, embed_plot, pred
