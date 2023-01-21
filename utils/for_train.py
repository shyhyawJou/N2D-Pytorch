from matplotlib import pyplot as plt
import pandas as pd

import torch
from torch import nn

from .nn import N2D
from .for_eval import accuracy


def train(model, opt, ds, device, epochs, n_cluster, save_dir):     
    print('begin train AutoEncoder ...')
    
    loss_fn = nn.MSELoss()
    n_sample, n_batch = len(ds.dataset), len(ds)
    model.train() 
    loss_h = History('min')
    
    for epoch in range(1, epochs + 1):
        print(f'\nEpoch {epoch}:')
        print('-' * 10)
        loss = 0.
        for i, (x, y) in enumerate(ds, 1):
            opt.zero_grad()
            x = x.to(device)
            _, gen = model(x)
            batch_loss = loss_fn(x, gen)
            batch_loss.backward()
            opt.step()
            loss += batch_loss * y.numel()
            print(f'{i}/{n_batch}', end='\r')

        loss /= n_sample
        loss_h.add(loss)
        if loss_h.better:
            torch.save(model, f'{save_dir}/AE.pt')
        print(f'loss : {loss.item():.4f}  min loss : {loss_h.best.item():.4f}')
        print(f'lr: {opt.param_groups[0]["lr"]}')

    df = pd.DataFrame(zip(range(1, epoch+1), loss_h.history), columns=['epoch', 'loss'])
    df.to_excel(f'{save_dir}/train.xlsx', index=False)
    
    print('\nload the best encoder to build N2D ...')
    model = torch.load(f'{save_dir}/AE.pt')
    model = N2D(model.encoder, n_cluster).to(device)
    acc, embedding, pred = accuracy(model, ds, device)
    print(f'train acc: {acc:.4f}')
    plot(embedding, pred, save_dir, acc, is_test=False)

    torch.save(model, f'{save_dir}/N2D.pt')
    
               
class History:
    def __init__(self, target='min'):
        self.value = None
        self.best = float('inf') if target == 'min' else 0.
        self.n_no_better = 0
        self.better = False
        self.target = target
        self.history = [] 
        self._check(target)
        
    def add(self, value):
        if self.target == 'min' and value < self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        elif self.target == 'max' and value > self.best:
            self.best = value
            self.n_no_better = 0
            self.better = True
        else:
            self.n_no_better += 1
            self.better = False
            
        self.value = value
        self.history.append(value.item())
        
    def _check(self, target):
        if target not in {'min', 'max'}:
            raise ValueError('target only allow "max" or "min" !')
    

def plot(embedding, cluster, save_dir, acc, is_test=True):
    print('plotting ...')

    plt.scatter(embedding[:, 0], embedding[:, 1], 16, cluster, cmap='Paired')
    if is_test:
        plt.title(f'Test data\nACC: {acc:.4f}')
        plt.savefig(f'{save_dir}/test.png')
    else:
        plt.title(f'training data\nACC: {acc:.4f}')
        plt.savefig(f'{save_dir}/train.png')
    plt.close()