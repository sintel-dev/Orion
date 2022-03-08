"""Primitive GANF

This primitive is an pytorch implementation of "Graph-Augmented 
Normalizing Flows for Anomaly Detection of Multiple Time Series" 
https://arxiv.org/pdf/2202.07857.pdf

This is a modified version of the original code, which can be found 
at https://github.com/EnyanDai/GANF/blob/main/models/GANF.py
"""
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.utils import clip_grad_value_
from torch.utils.data import Dataset, DataLoader

from orion.primitives.nf import MAF, RealNVP


class Signal(Dataset):
    def __init__(self, df, label=None, window_size=60, stride_size=10, unit='s'):
        super(Signal, self).__init__()
        if label is None:
            self.has_labels = False
        else:
            self.has_labels = True

        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size
        self.unit = unit
        self.data, self.idx, self.label = self.preprocess(df, label)
    
    def preprocess(self, df, label):
        start_idx = np.arange(0, len(df) - self.window_size, self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)

        delta_time = df.index[end_idx] - df.index[start_idx]
        idx_mask = delta_time == pd.Timedelta(self.window_size, unit=self.unit)

        if self.has_labels:
            return df.values, start_idx[idx_mask], label[start_idx[idx_mask]]
        
        return df.values, start_idx[idx_mask], None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        #  N X K X L X D 
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])

        if self.has_labels:
            return torch.FloatTensor(data).transpose(0, 1), self.label[index]

        return torch.FloatTensor(data).transpose(0, 1)


class GNN(nn.Module):
    """
    The GNN module applied in GANF
    """
    def __init__(self, input_size, hidden_size):

        super(GNN, self).__init__()
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, A):
        ## A: K X K
        ## H: N X K  X L X D

        h_n = self.lin_n(torch.einsum('nkld,kj->njld',h,A))
        h_r = self.lin_r(h[:,:,:-1])
        h_n[:,:,1:] += h_r
        h = self.lin_2(F.relu(h_n))

        return h


class GANF(nn.Module):
    def __init__ (self, n_blocks, input_size, hidden_size, n_hidden, dropout=0.1, model="MAF", batch_norm=True):
        super(GANF, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size,hidden_size=hidden_size,batch_first=True, dropout=dropout)
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size)
        if model=="MAF":
            self.nf = MAF(n_blocks, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm,activation='tanh')
        else:
            self.nf = RealNVP(n_blocks, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm)

    def forward(self, x, A):

        return self.test(x, A).mean()

    def test(self, x, A):
        # x: N X K X L X D 
        full_shape = x.shape

        # reshape: N*K, L, D
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        h,_ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))


        h = self.gcn(h, A)

        # reshappe N*K*L,H
        h = h.reshape((-1,h.shape[3]))
        x = x.reshape((-1,full_shape[3]))

        log_prob = self.nf.log_prob(x,h).reshape([full_shape[0],-1])#*full_shape[1]*full_shape[2]
        log_prob = log_prob.mean(dim=1)

        return log_prob
    
    def locate(self, x, A):
        # x: N X K X L X D 
        full_shape = x.shape

        # reshape: N*K, L, D
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        h,_ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))


        h = self.gcn(h, A)

        # reshappe N*K*L,H
        h = h.reshape((-1,h.shape[3]))
        x = x.reshape((-1,full_shape[3]))

        log_prob = self.nf.log_prob(x,h).reshape([full_shape[0],full_shape[1],-1])#*full_shape[1]*full_shape[2]
        log_prob = log_prob.mean(dim=2)

        return log_prob

def _fit(model, train_loader, valid_loader, optimizer, iteration, n_epochs, A, channels, rho, 
         alpha, save_path, device, verbose, loss_best=None, log_interval=None):
    loss_best = loss_best or False
    log_interval = log_interval or False

    for epoch in range(n_epochs):
        print(epoch)
        loss_train = []
        model.train()
        for x in train_loader:
            x = x.to(device)

            optimizer.zero_grad()
            loss = -model(x,  A)
            h = torch.trace(torch.matrix_exp(A * A)) - channels
            total_loss = loss + 0.5 * rho * h * h + alpha * h

            total_loss.backward()
            clip_grad_value_(model.parameters(), 1)
            optimizer.step()
            loss_train.append(loss.item())
            A.data.copy_(torch.clamp(A.data, min=0, max=1))
            
        print("evaluate")
        # evaluate
        loss_val = 0
        if valid_loader is not None:
            model.eval()
            loss_val = []
            with torch.no_grad():
                for x in valid_loader:
                    x = x.to(device)
                    loss = -model.test(x, A.data).cpu().numpy()
                    loss_val.append(loss)

            loss_val = np.concatenate(loss_val)
            loss_val = np.nan_to_num(loss_val)

            if verbose and labels:
                roc_val = roc_auc_score(
                    np.asarray(val_loader.dataset.label.values, dtype=int), loss_val)

                print('[{}] Epoch: {}/{}, valid ROC AUC: {}'.format(
                    iteration, epoch, n_epochs, roc_val))

        if verbose:
            # will print 0 for valid if not initiated
            print('[{}] Epoch: {}/{}, train -log_prob: {:.2f}, valid -log_prob: {:.2f}'.format(
                iteration, epoch, n_epochs, np.mean(loss_train), np.mean(loss_val)))

    if verbose:
        print('rho: {}, alpha {}, h {}'.format(rho, alpha, h.item()))


    if loss_best and np.mean(loss_val) < loss_best:
        loss_best = np.mean(loss_val)
        print("save model {} epoch".format(epoch))
        torch.save(A.data,os.path.join(save_path, "graph_best.pt"))
        torch.save(model.state_dict(), os.path.join(save_path, "{}_best.pt".format(args.name)))

    if log_interval and epoch % log_interval==0:
        torch.save(A.data,os.path.join(save_path, "graph_{}.pt".format(epoch)))
        torch.save(model.state_dict(), os.path.join(save_path, "{}_{}.pt".format(name, epoch)))

    return h

def fit(data, timestamp_column='timestamp', target_column='label', valid_split=0.2, seed=18, 
        n_blocks=1, n_components=1, hidden_size=32, n_hidden=1, batch_norm=True, batch_size=512, 
        dropout=0.0, weight_decay=5e-4, n_epochs=1, lr=2e-3, log_interval=5, model_path=None,
        output_dir='./checkpoint/model', name='ganf', verbose=True):
    """Train GANF model.

    Args:
        * data (pands.DataFrame): 
            A dataframe with ``timestamp`` and feature columns used for training. 
        * timestamp_column (str):
            Name of the ``timestamp`` column.
        * target_column (str):
            Name of the ``label`` column.
        * valid_split (float): 
            A float to split data dataframe to validation set. Data needs to contain a label
            column. Use ``target_column`` to change the target column name.
        * seed (int):
            Random seed to fix in training.
        * n_blocks (int):
            Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).
        * n_components (int):
            Number of Gaussian clusters for mixture of gaussians models.
        * hidden_size (int):
            Hidden layer size for MADE (and each MADE block in an MAF).
        * n_hidden (int):
            Number of hidden layers in each MADE.
        * batch_norm (bool):
            Whether to use batch norm or not.
        * batch_size (int):
            Number of example per epoch.
        * weight_decay (float):
            Weight decay rate for Adam.
        * n_epochs (int):
            Number of iterations to train the model.
        * lr (float):
            Learning rate.
        * log_interval (int):
            How often to show loss statistics and save samples.
    """

    # ------------------------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------------------------
    print("preparing data")

    data[timestamp_column] = pd.to_datetime(data[timestamp_column])
    data = data.set_index(timestamp_column)

    labels = None
    if target_column in data.columns:
        labels = data.pop(target_column)
        
    # normalize data
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    features = (data - mean) / std
    features = features.dropna(axis=1)
    channels = len(features.columns)

    train = features
    valid_loader = None
    # split data
    if valid_split > 0:
        valid_size = int(len(data) * valid_split)
        train = features.iloc[: -valid_size]
        valid = features.iloc[-valid_size: ]

        if labels is not None:
            labels = labels.iloc[-valid_size: ]

        valid_loader = DataLoader(Signal(valid, labels), batch_size=batch_size, shuffle=False)

    train_loader = DataLoader(Signal(train), batch_size=batch_size, shuffle=True)

    # ------------------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------------------
    print("seeding")
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # ------------------------------------------------------------------------------
    # Set model hyperparameters
    # ------------------------------------------------------------------------------
    print("preparing hyperparameters")

    rho = 1.0
    alpha = 0.0
    lambda1 = 0.0
    h_A_old = np.inf
    max_iter = 20
    rho_max = 1e16
    h_tol = 1e-4
    epoch = 0

    init = torch.zeros([channels, channels])
    init = xavier_uniform_(init).abs()
    init = init.fill_diagonal_(0.0)
    A = torch.tensor(init, requires_grad=True, device=device)

    # ------------------------------------------------------------------------------
    # Load and instantiate model
    # ------------------------------------------------------------------------------
    print("preparing model")

    model = GANF(n_blocks, 1, hidden_size, n_hidden, dropout=dropout, batch_norm=batch_norm)
    model = model.to(device)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        print('Load model from '+ model_path)

    save_path = os.path.join(output_dir, name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # ------------------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------------------
    print("starting training")

    loss_best = np.inf
    for iteration in range(max_iter):
        while rho < rho_max:
            optimizer = torch.optim.Adam([
                {'params':model.parameters(), 'weight_decay': weight_decay},
                {'params': [A]}], lr=lr, weight_decay=0.0)
            
            h = _fit(
                model=model, 
                train_loader=train_loader, 
                valid_loader=valid_loader,
                optimizer=optimizer,
                iteration=iteration, 
                n_epochs=n_epochs, 
                A=A, 
                channels=channels, 
                rho=rho, 
                alpha=alpha, 
                save_path=save_path,
                device=device,
                verbose=verbose
            )

            del optimizer
            torch.cuda.empty_cache()
            
            if h.item() > 0.5 * h_A_old:
                rho *= 10
            else:
                break


        h_A_old = h.item()
        alpha += rho*h.item()

        if h_A_old <= h_tol or rho >=rho_max:
            break

    optimizer = torch.optim.Adam([
        {'params':model.parameters(), 'weight_decay': weight_decay},
        {'params': [A]}], lr=lr, weight_decay=0.0)

    _fit(
        model=model, 
        train_loader=train_loader, 
        valid_loader=valid_loader,
        optimizer=optimizer,
        iteration='final', 
        n_epochs=30, 
        A=A, 
        channels=channels, 
        rho=rho, 
        alpha=alpha, 
        save_path=save_path,
        device=device,
        verbose=verbose,
        loss_best=loss_best, 
        log_interval=loss_best
    )

