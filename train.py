import argparse
import os
from pathlib import Path as p
from time import time
import numpy as np
import random

import torch
from torch.optim import AdamW

from utils import (load_data, plot,
                   AutoEncoder,
                   train, 
                   accuracy)



def get_arg():
    arg = argparse.ArgumentParser()
    arg.add_argument('-bs', default=256, type=int, help='batch size')
    arg.add_argument('-epoch', type=int, help='epochs for train DEC')
    arg.add_argument('-k', type=int, help='num of clusters')
    arg.add_argument('-save_dir', default='weight', help='location where model will be saved')
    arg.add_argument('-worker', default=4, type=int, help='num of workers')
    arg.add_argument('-seed', type=int, default=None, help='torch random seed')
    arg = arg.parse_args()
    return arg
    

def main():
    arg = get_arg()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not os.path.exists(arg.save_dir):
        os.makedirs(arg.save_dir, exist_ok=True) 
    else:
        for path in p(arg.save_dir).glob('*.png'):
            path.unlink()
        
    if arg.seed is not None:
        random.seed(arg.seed)
        np.random.seed(arg.seed)
        torch.manual_seed(arg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    tr_ds, test_ds = load_data(arg.bs, arg.worker)
         
    print('\ntrain num:', len(tr_ds.dataset))
    print('test num:', len(test_ds.dataset))
        
    # train autoencoder
    ae = AutoEncoder().to(device)    
    print(f'\nAE param: {sum(p.numel() for p in ae.parameters()) / 1e6:.2f} M')
    opt = AdamW(ae.parameters())
    t0 = time()
    train(ae, opt, tr_ds, device, arg.epoch, arg.k, arg.save_dir)
    t1 = time()
        
    # Evaluate
    print()
    print('*' * 50)
    print('Evaluate test data ...')
    n2d = torch.load(f'{arg.save_dir}/N2D.pt', device)
    acc, embedding, pred = accuracy(n2d, test_ds, device)
    print(f'test acc: {acc:.4f}')
    plot(embedding, pred, arg.save_dir, acc, is_test=True)
    
    print(f'\ntrain AE time: {t1 - t0:.2f} s')


    
if __name__ == '__main__':
    main()
