import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
from utils import *
from metrics import *
from model import SocialImplicit
from amd_amv_kde_metrics import calc_amd_amv, kde_lossf
from CFG import CFG

# Determinar dtype dinámicamente para compatibilidad
def get_device_and_dtype():
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dt = torch.float32 if dev.type == 'mps' else torch.float64
    return dev, dt

def test(loader_test, model, device, ROBUSTNESS, KSTEPS=20):
    model.eval()
    ade_bigls = []
    fde_bigls = []
    mabs_loss = []
    kde_loss = []
    eig_collect = []
    
    for batch in loader_test:
        # Get data and move to the dynamically selected device (CUDA/MPS/CPU)
        batch = [tensor.to(device=device, dtype=dtype) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        num_of_objs = obs_traj_rel.shape[1]
        V_tr = V_tr.squeeze()
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.detach().cpu().numpy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(
            V_obs.detach().cpu().numpy().squeeze(), V_x[0, :, :].copy())

        V_y = seq_to_nodes(pred_traj_gt.detach().cpu().numpy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(
            V_tr.detach().cpu().numpy().squeeze(), V_x[-1, :, :].copy())

        for n in range(num_of_objs):
            ade_ls[n] = []
            fde_ls[n] = []

        V_predx = model(V_obs_tmp, obs_traj, KSTEPS=KSTEPS)

        b_samples = []
        for k in range(KSTEPS):
            V_pred = V_predx[k:k + 1, ...]
            V_pred = V_pred.permute(0, 2, 3, 1)
            V_pred = V_pred.squeeze()

            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(
                V_pred.detach().cpu().numpy().squeeze(), V_x[-1, :, :].copy())

            # Sensitivity adjustment
            V_pred_rel_to_abs += ROBUSTNESS  # 0.01 = 1 cm, 0.1 = 10 cm

            b_samples.append(V_pred_rel_to_abs[:, :, None, :].copy())

            for n in range(num_of_objs):
                pred = []
                target = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n:n + 1, :])
                target.append(V_y_rel_to_abs[:, n:n + 1, :])
                number_of.append(1)

                ade_ls[n].append(ade(pred, target, number_of))
                fde_ls[n].append(fde(pred, target, number_of))

        abs_samples = np.concatenate(b_samples, axis=2)  # ab samples in (12,3,100,2) gt in (12,3,2)

        m, _, _, _, eig = calc_amd_amv(V_y_rel_to_abs.copy(),
                                                  abs_samples.copy())
        mabs_loss.append(m)  
        eig_collect.append(eig)
        _kde = kde_lossf(V_y_rel_to_abs.copy(), abs_samples.copy())
        kde_loss.append(_kde)
        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

    ade_ = sum(ade_bigls) / len(ade_bigls) if len(ade_bigls) > 0 else 0
    fde_ = sum(fde_bigls) / len(fde_bigls) if len(fde_bigls) > 0 else 0

    return ade_, fde_, sum(kde_loss) / len(kde_loss), sum(mabs_loss) / len(mabs_loss), sum(eig_collect) / len(eig_collect)

# The Main Block acts as a shield to prevent multiprocessing spawn loops on Windows
if __name__ == '__main__':
    # Determine the hardware target and dtype globally
    device, dtype = get_device_and_dtype()
    print(f"Using hardware device: {device}")

    for ROBUSTNESS in [0]:  #, -0.1, -0.01, +0.01, +0.1]:
        print("*" * 30)
        print("*" * 30)
        print("ROBUSTNESS:", ROBUSTNESS)
        print("*" * 30)
        print("*" * 30)

        paths = [
            './checkpoint/social-implicit-eth',
            './checkpoint/social-implicit-hotel',
            './checkpoint/social-implicit-zara1',
            './checkpoint/social-implicit-zara2',
            './checkpoint/social-implicit-univ',
            './checkpoint/social-implicit-sdd',
        ]
        KSTEPS = 1000
        EASY_RESULTS = []

        print("*" * 50)
        print('Number of samples:', KSTEPS)
        print("*" * 50)

        for feta in range(len(paths)):
            ade_ls = []
            fde_ls = []
            exp_ls = []
            kde_ls = []
            amd_ls = []
            eig_ls = []
            path = paths[feta]
            exps = glob.glob(path)
            exps.sort()

            for exp_path in exps:
                model_path = exp_path + '/val_best.pth'
                args_path = exp_path + '/args.pkl'
                
                if not os.path.exists(args_path) or not os.path.exists(model_path):
                    continue
                
                with open(args_path, 'rb') as f:
                    args = pickle.load(f)

                stats = exp_path + '/constant_metrics.pkl'
                if os.path.exists(stats):
                    with open(stats, 'rb') as f:
                        cm = pickle.load(f)

                # Data prep
                obs_seq_len = args.obs_seq_len
                pred_seq_len = args.pred_seq_len
                data_set = './datasets/' + args.dataset + '/'

                dset_test = TrajectoryDataset(data_set + 'test/',
                                              obs_len=obs_seq_len,
                                              pred_len=pred_seq_len,
                                              skip=1,
                                              norm_lap_matr=True)

                loader_test = DataLoader(
                    dset_test,
                    batch_size=1,  
                    shuffle=False,
                    num_workers=1) # Kept at 0 for stability, safe to increase now if desired

                # Defining the model parameters
                is_eth = args.dataset == 'eth'
                if is_eth:
                    noise_weight = CFG["noise_weight_eth"]
                else:
                    noise_weight = CFG["noise_weight"]

                # Initialize model and map it to the correct device dynamically
                model = SocialImplicit(spatial_input=CFG["spatial_input"],
                                       spatial_output=CFG["spatial_output"],
                                       temporal_input=CFG["temporal_input"],
                                       temporal_output=CFG["temporal_output"],
                                       bins=CFG["bins"],
                                       noise_weight=noise_weight).to(device)

                # Safely load weights and force model tensor types
                model.load_state_dict(torch.load(model_path, map_location=device))
                model = model.to(device=device, dtype=dtype)
                model.eval()

                ade_ = 999999
                fde_ = 999999
                
                # Execute test passing local variables instead of relying on globals
                ad, fd, kd, md, eg = test(loader_test, model, device, ROBUSTNESS, KSTEPS=KSTEPS)
                
                ade_ = min(ade_, ad)
                fde_ = min(fde_, fd)
                ade_ls.append(ade_)
                fde_ls.append(fde_)
                kde_ls.append(kd)
                amd_ls.append(md)
                exp_ls.append(exp_path)
                eig_ls.append(eg)
                print("amd,kde,amv:", md, kd, eg)
            
            print("*" * 50)

            if len(ade_ls) > 0:
                ade_ls = np.asarray(ade_ls)
                fde_ls = np.asarray(fde_ls)
                kde_ls = np.asarray(kde_ls)
                amd_ls = np.asarray(amd_ls)
                eig_ls = np.asarray(eig_ls)

                avg_eig_mde = (eig_ls + amd_ls) / 2.0
                min_avg_eig_ade = np.argmin(avg_eig_mde)

                EASY_RESULTS.append([
                    exp_ls[min_avg_eig_ade],
                    round(amd_ls[min_avg_eig_ade], 4),
                    round(kde_ls[min_avg_eig_ade], 4), eig_ls[min_avg_eig_ade]
                ])

        for kkkk in EASY_RESULTS:
            print(kkkk)