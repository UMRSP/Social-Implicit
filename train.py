import os
import math
import sys
import pickle
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from metrics import *
from model import SocialImplicit
from trajectory_augmenter import TrajectoryAugmenter
from CFG import CFG

# Determine the best available hardware globally
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# MPS does not support float64 (double)
dtype = torch.float32 if device.type == 'mps' else torch.float64

def cdist_cosine_sim(a, b, eps=1e-08):
    a_norm = a / torch.clamp(a.norm(dim=1)[:, None], min=eps)
    b_norm = b / torch.clamp(b.norm(dim=1)[:, None], min=eps)
    return torch.acos(
        torch.clamp(torch.mm(a_norm, b_norm.transpose(0, 1)),
                    min=-1.0 + eps,
                    max=1.0 - eps))

def implicit_likelihood_estimation(V_pred, V_target, args, loss_store):
    l1_mean = nn.L1Loss()
    V_pred = V_pred.contiguous()
    diff = torch.abs(V_pred - V_target)
    diff_sum = torch.sum(diff, dim=(1, 2, 3))
    _, indices = torch.sort(diff_sum)
    min_indx = indices[0]
    V_pred_min = V_pred[min_indx]
    V_target_sq = V_target.squeeze()

    error = l1_mean(V_pred_min, V_target_sq)
    trip_loss = l1_mean(V_pred_min, V_pred[indices[1]]) - l1_mean(V_pred_min, V_pred[indices[-1]])

    V_pred_min_ = V_pred_min.reshape(-1, 2)
    V_target_ = V_target_sq.reshape(-1, 2)

    norm_loss = torch.abs(
        torch.cdist(V_pred_min_.unsqueeze(0), V_pred_min_.unsqueeze(0), p=2.0)
        - torch.cdist(V_target_.unsqueeze(0), V_target_.unsqueeze(0), p=2.0)
    ).mean()

    cos_loss = torch.abs(
        cdist_cosine_sim(V_pred_min_, V_pred_min_) -
        cdist_cosine_sim(V_target_, V_target_)).mean()

    loss_store["l2"] += error.item()
    loss_store["gl2"] += norm_loss.item()
    loss_store["gcos"] += cos_loss.item()
    loss_store["trip"] += trip_loss.item()

    return error + args.w_norm * norm_loss + args.w_trip * trip_loss + args.w_cos * cos_loss

def train(epoch, model, loader_train, optimizer, metrics, args, trajaugmenter):
    model.train()
    loss_store = {"l2": 0, "gl2": 0, "gcos": 0, "trip": 0}
    total_loss = 0
    batch_loss = 0
    
    for cnt, batch in enumerate(loader_train):
        # Unpack and move to device
        batch = [tensor.to(device=device, dtype=dtype) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        # Augment
        V_obs, V_tr, obs_traj, pred_traj_gt = trajaugmenter.augment(V_obs, V_tr, obs_traj, pred_traj_gt)
        
        optimizer.zero_grad()
        
        # Forward
        V_pred = model(V_obs.permute(0, 3, 1, 2), obs_traj)
        V_pred = V_pred.permute(0, 2, 3, 1)

        # Loss
        loss = implicit_likelihood_estimation(V_pred, V_tr, args, loss_store)
        batch_loss += loss
        total_loss += loss.item()

        # Update
        if cnt % args.batch_size == 0 and cnt != 0:
            (batch_loss / args.batch_size).backward()
            
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
            optimizer.step()
            
            print(f"{args.tag} | TRAIN | Epoch: {epoch} | Batch loss: {batch_loss.item()/args.batch_size:.4f}")
            print("Detailed train loss:", {k: v/args.batch_size for k, v in loss_store.items()})
            
            # Reset
            batch_loss = 0
            loss_store = {"l2": 0, "gl2": 0, "gcos": 0, "trip": 0}
            
    metrics['train_loss'].append(total_loss / (cnt + 1))

def vald(epoch, model, loader_val, metrics, constant_metrics, checkpoint_dir, args):
    model.eval()
    total_loss = 0
    loss_store = {"l2": 0, "gl2": 0, "gcos": 0, "trip": 0}
    
    with torch.no_grad():
        for cnt, batch in enumerate(loader_val):
            batch = [tensor.to(device=device, dtype=dtype) for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch
            
            V_pred = model(V_obs.permute(0, 3, 1, 2), obs_traj)
            V_pred = V_pred.permute(0, 2, 3, 1)
            
            total_loss += implicit_likelihood_estimation(V_pred, V_tr, args, loss_store).item()

        avg_loss = total_loss / (cnt + 1)
        metrics['val_loss'].append(avg_loss)
        
        print(f"{args.tag} | VALD | Epoch: {epoch} | Loss: {avg_loss:.4f}")
        
        # Save best model
        store_per = 0.05 * constant_metrics['min_val_loss']
        if (constant_metrics['min_val_loss'] - avg_loss) > store_per:
            constant_metrics['min_val_loss'] = avg_loss
            constant_metrics['min_val_epoch'] = epoch
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'val_best.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ... (args definition same as yours)
    parser.add_argument('--w_norm', type=float, default=0.0001)
    parser.add_argument('--w_cos', type=float, default=0.0001)
    parser.add_argument('--w_trip', type=float, default=0.0001)
    parser.add_argument('--obs_seq_len', type=int, default=8)
    parser.add_argument('--pred_seq_len', type=int, default=12)
    parser.add_argument('--dataset', default='hotel')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--clip_grad', type=float, default=None)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_sh_rate', type=int, default=45)
    parser.add_argument('--tag', default='tag')
    args = parser.parse_args()

    print(f"Using device: {device}")
    
    # Data Prep
    dset_train = TrajectoryDataset(f'./datasets/{args.dataset}/train/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, norm_lap_matr=True)
    loader_train = DataLoader(dset_train, batch_size=1, shuffle=True, num_workers=1)

    dset_val = TrajectoryDataset(f'./datasets/{args.dataset}/val/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, norm_lap_matr=True)
    loader_val = DataLoader(dset_val, batch_size=1, shuffle=False, num_workers=1)

    # Model
    noise_weight = CFG["noise_weight_eth"] if args.dataset == 'eth' else CFG["noise_weight"]
    model = SocialImplicit(spatial_input=CFG["spatial_input"], spatial_output=CFG["spatial_output"],
                           temporal_input=CFG["temporal_input"], temporal_output=CFG["temporal_output"],
                           bins=CFG["bins"], noise_weight=noise_weight).to(device=device, dtype=dtype)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.1)
    
    checkpoint_dir = './checkpoint/' + args.tag + '/'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    metrics = {'train_loss': [], 'val_loss': []}
    constant_metrics = {'min_val_epoch': -1, 'min_val_loss': float('inf')}
    trajaugmenter = TrajectoryAugmenter(data_loader=loader_train)

    for epoch in range(args.num_epochs):
        train(epoch, model, loader_train, optimizer, metrics, args, trajaugmenter)
        vald(epoch, model, loader_val, metrics, constant_metrics, checkpoint_dir, args)
        scheduler.step()
        
        # Logging ...
        with open(checkpoint_dir + 'metrics.pkl', 'wb') as fp: pickle.dump(metrics, fp)
        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp: pickle.dump(constant_metrics, fp)