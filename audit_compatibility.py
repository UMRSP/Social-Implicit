import sys
import os
import torch
import numpy as np

# Import components from local project
try:
    from model import SocialImplicit
    from trajectory_augmenter import TrajectoryAugmenter
    from CFG import CFG
    print("[SUCCESS] Successfully imported model, trajectory_augmenter, and CFG.")
except ImportError as e:
    print(f"[ERROR] Failed to import local project modules: {e}")
    sys.exit(1)

def run_device_diagnostics():
    print("=" * 60)
    print("      SYSTEM & PYTORCH ENVIRONMENT DIAGNOSTICS      ")
    print("=" * 60)
    print(f"OS Platform: {sys.platform}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"  - CUDA Device Count: {torch.cuda.device_count()}")
        print(f"  - Current Device: {torch.cuda.current_device()}")
        print(f"  - Device Name: {torch.cuda.get_device_name(0)}")
        print(f"  - PyTorch CUDA Build: {torch.version.cuda}")

    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    print(f"MPS (Apple Silicon Metal) Available: {mps_available}")
    
    cpu_info = "Available"
    print(f"CPU Status: {cpu_info}")
    print("=" * 60)
    
    return {
        "cuda": cuda_available,
        "mps": mps_available,
        "cpu": True
    }

def test_model_forward_pass(device_name, dtype):
    """Test a forward pass on SocialImplicit with a specific device and dtype."""
    device = torch.device(device_name)
    print(f"Testing model on [{device_name}] with dtype [{dtype}]...")
    
    try:
        # Create dummy inputs resembling batch from loader
        # obs_traj shape: (batch_size, num_objs, seq_len) -> e.g. (1, 3, 8, 2)
        # v shape: (batch_size, channels, seq_len, num_objs) -> (1, 2, 8, 3)
        batch_size = 1
        num_objs = 3
        seq_len = 8
        
        v = torch.randn(batch_size, 2, seq_len, num_objs, device=device, dtype=dtype)
        obs_traj = torch.randn(batch_size, num_objs, 2, seq_len, device=device, dtype=dtype)
        
        # Initialize model and put on device + dtype
        noise_weight = CFG["noise_weight"]
        model = SocialImplicit(
            spatial_input=CFG["spatial_input"],
            spatial_output=CFG["spatial_output"],
            temporal_input=CFG["temporal_input"],
            temporal_output=CFG["temporal_output"],
            bins=CFG["bins"],
            noise_weight=noise_weight
        ).to(device=device, dtype=dtype)
        
        model.eval()
        with torch.no_grad():
            output = model(v, KSTEPS=20)
            
        print(f"  [PASS] Forward pass successful. Output shape: {output.shape}")
        return True, None
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        return False, str(e)

def test_data_augmentation(device_name, dtype):
    """Test TrajectoryAugmenter functions under the given device and dtype."""
    device = torch.device(device_name)
    print(f"Testing TrajectoryAugmenter on [{device_name}] with dtype [{dtype}]...")
    
    try:
        augmenter = TrajectoryAugmenter(total_time=20, split_time=8, data_loader=None)
        
        # obs_traj: [1, 2, 2, 8], pred_traj_gt: [1, 2, 2, 12]
        obs_traj = torch.randn(1, 2, 2, 8, device=device, dtype=dtype)
        pred_traj_gt = torch.randn(1, 2, 2, 12, device=device, dtype=dtype)
        
        # Test individual augmentation methods
        print("  - Testing _aug_jitter...")
        j_obs, j_tr, j_obs_t, j_pred_t = augmenter._aug_jitter(obs_traj, pred_traj_gt)
        assert j_obs.dtype == dtype, f"Expected {dtype}, got {j_obs.dtype}"
        
        print("  - Testing _aug_flip_mirror...")
        fm_obs, fm_tr, fm_obs_t, fm_pred_t = augmenter._aug_flip_mirror(obs_traj, pred_traj_gt)
        assert fm_obs.dtype == dtype
        
        print("  - Testing _aug_flip_reverse...")
        fr_obs, fr_tr, fr_obs_t, fr_pred_t = augmenter._aug_flip_reverse(obs_traj, pred_traj_gt)
        assert fr_obs.dtype == dtype

        print("  - Testing _aug_speed...")
        s_obs, s_tr, s_obs_t, s_pred_t = augmenter._aug_speed(obs_traj, pred_traj_gt)
        assert s_obs.dtype == dtype

        print("  - Testing _aug_rot (rotation matrix integration)...")
        r_obs, r_tr, r_obs_t, r_pred_t = augmenter._aug_rot(obs_traj, pred_traj_gt)
        assert r_obs.dtype == dtype, f"Expected {dtype}, got {r_obs.dtype}"

        print("  [PASS] All core augmentations passed successfully.")
        return True, None
    except Exception as e:
        print(f"  [FAIL] Augmentation failed: {e}")
        return False, str(e)

def main():
    backends = run_device_diagnostics()
    
    devices_to_test = []
    if backends["cuda"]:
        devices_to_test.append("cuda")
    if backends["mps"]:
        devices_to_test.append("mps")
    devices_to_test.append("cpu")
    
    report = []
    has_failures = False
    
    print("\n" + "=" * 60)
    print("            RUNNING COMPATIBILITY AUDITS            ")
    print("=" * 60)
    
    for dev in devices_to_test:
        # MPS only supports Float32 properly in standard PyTorch layers
        dtypes = [torch.float32] if dev == "mps" else [torch.float32, torch.float64]
        
        for dt in dtypes:
            # 1. Test model compatibility
            m_ok, m_err = test_model_forward_pass(dev, dt)
            report.append({
                "component": "SocialImplicit Model",
                "device": dev,
                "dtype": str(dt),
                "status": "PASS" if m_ok else "FAIL",
                "error": m_err
            })
            if not m_ok:
                has_failures = True
                
            # 2. Test data augmentation compatibility
            a_ok, a_err = test_data_augmentation(dev, dt)
            report.append({
                "component": "TrajectoryAugmenter",
                "device": dev,
                "dtype": str(dt),
                "status": "PASS" if a_ok else "FAIL",
                "error": a_err
            })
            if not a_ok:
                has_failures = True
            print("-" * 60)

    # Print Final Summary Report
    print("\n" + "=" * 60)
    print("                 COMPATIBILITY AUDIT REPORT                 ")
    print("=" * 60)
    print(f"{'Component':<22} | {'Device':<6} | {'Dtype':<13} | {'Status':<6}")
    print("-" * 60)
    for entry in report:
        print(f"{entry['component']:<22} | {entry['device']:<6} | {entry['dtype']:<13} | {entry['status']:<6}")
    print("=" * 60)
    
    if has_failures:
        print("\n[WARNING] Compatibility audit found potential errors/warnings.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] Compatibility audit passed flawlessly across tested device/dtype combinations!")
        sys.exit(0)

if __name__ == "__main__":
    main()
