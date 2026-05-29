import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import pickle
import glob
from utils import *
from metrics import *
from model import SocialImplicit
from CFG import CFG

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

device = get_device()
dtype = torch.float32 if device.type == 'mps' else torch.float64

def test(loader_test, model, device, ROBUSTNESS, KSTEPS=20):
    model.eval()
    ade_bigls = []
    fde_bigls = []

    for batch in loader_test:
        # Get data and dynamically send to the available hardware (CUDA/MPS/CPU)
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

        V_predx = model(V_obs_tmp, KSTEPS=KSTEPS)

        for k in range(KSTEPS):
            V_pred = V_predx[k:k + 1, ...]
            V_pred = V_pred.permute(0, 2, 3, 1)
            V_pred = V_pred.squeeze()

            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(
                V_pred.detach().cpu().numpy().squeeze(), V_x[-1, :, :].copy())
                
            # Sensitivity
            V_pred_rel_to_abs += ROBUSTNESS

            for n in range(num_of_objs):
                pred = []
                target = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n:n + 1, :])
                target.append(V_y_rel_to_abs[:, n:n + 1, :])
                number_of.append(1)

                ade_ls[n].append(ade(pred, target, number_of))
                fde_ls[n].append(fde(pred, target, number_of))

        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

    ade_ = sum(ade_bigls) / len(ade_bigls) if len(ade_bigls) > 0 else 0
    fde_ = sum(fde_bigls) / len(fde_bigls) if len(fde_bigls) > 0 else 0
    
    return ade_, fde_


# Main block acts as a shield against Windows multiprocessing spawn loops
if __name__ == '__main__':
    # Determine the best available hardware globally
    device = get_device()
    print(f"Using hardware device: {device}")


    for ROBUSTNESS in [0]:  # [-0.1, -0.01, 0, +0.01, +0.1]:
        print("\n" + "*" * 30)
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
        KSTEPS = 20

        EASY_RESULTS = []

        print("*" * 50)
        print('Number of samples:', KSTEPS)
        print("*" * 50)

        for feta in range(len(paths)):

            ade_ls = []
            fde_ls = []
            exp_ls = []
            path = paths[feta]
            exps = glob.glob(path)
            exps.sort()
            print('Models being tested are:', exps)

            for exp_path in exps:

                print("*" * 50)
                print("Evaluating model:", exp_path)

                model_path = exp_path + '/val_best.pth'
                args_path = exp_path + '/args.pkl'
                
                # Check if paths exist to avoid crashing on incomplete folders
                if not os.path.exists(args_path) or not os.path.exists(model_path):
                    print(f"Skipping {exp_path}, missing args or weights.")
                    continue
                
                with open(args_path, 'rb') as f:
                    args = pickle.load(f)

                stats = exp_path + '/constant_metrics.pkl'
                if os.path.exists(stats):
                    with open(stats, 'rb') as f:
                        cm = pickle.load(f)
                    print("Stats:", cm)

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
                    batch_size=1,  # Irrelative to the args batch size parameter
                    shuffle=False,
                    num_workers=1) # Safe to keep at 0, or increase if you want multiprocessing

                # Defining the model
                is_eth = args.dataset == 'eth'
                if is_eth:
                    noise_weight = CFG["noise_weight_eth"]
                else:
                    noise_weight = CFG["noise_weight"]

                # Initialize and map the model dynamically to the device
                model = SocialImplicit(spatial_input=CFG["spatial_input"],
                                       spatial_output=CFG["spatial_output"],
                                       temporal_input=CFG["temporal_input"],
                                       temporal_output=CFG["temporal_output"],
                                       bins=CFG["bins"],
                                       noise_weight=noise_weight).to(device)
                                       
                # Load weights mapping them to the assigned device
                model.load_state_dict(torch.load(model_path, map_location=device))
                model = model.to(device=device, dtype=dtype)
                model.eval()

                ade_ = 999999
                fde_ = 999999
                print("Testing ....")
                
                # Pass local parameters instead of globals
                ad, fd = test(loader_test, model, device, ROBUSTNESS, KSTEPS=KSTEPS)
                
                ade_ = min(ade_, ad)
                fde_ = min(fde_, fd)
                ade_ls.append(ade_)
                fde_ls.append(fde_)
                exp_ls.append(exp_path)
                print("ADE:", ade_, " FDE:", fde_)
                
            print("*" * 50)
            
            if len(ade_ls) > 0:
                ade_ls = np.asarray(ade_ls)
                fde_ls = np.asarray(fde_ls)
                avg_ade_fde = (ade_ls + fde_ls) / 2.0
                min_avg_ade_fde = np.argmin(avg_ade_fde)

                EASY_RESULTS.append([
                    exp_ls[min_avg_ade_fde],
                    round(ade_ls[min_avg_ade_fde], 4),
                    round(fde_ls[min_avg_ade_fde], 4)
                ])
                
        print(EASY_RESULTS)