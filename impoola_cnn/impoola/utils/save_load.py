import os

import torch


def save_checkpoint(agent, optimizer, args, global_step, envs, output_dir, run_name, checkpoint_name):
    """Save model checkpoint"""
    checkpoint_path = os.path.join(output_dir, f"{run_name}_{checkpoint_name}.pt")
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'global_step': global_step,
        'obs_rms': getattr(envs, 'obs_rms', None),
        'return_rms': getattr(envs, 'return_rms', None),
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(agent, optimizer, checkpoint_path, device, envs=None):
    """Load model checkpoint"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(checkpoint['agent_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    global_step = checkpoint.get('global_step', 0)

    if envs is not None:
        if 'obs_rms' in checkpoint and checkpoint['obs_rms'] is not None:
            envs.obs_rms = checkpoint['obs_rms']
        if 'return_rms' in checkpoint and checkpoint['return_rms'] is not None:
            envs.return_rms = checkpoint['return_rms']

    print(f"Checkpoint loaded: {checkpoint_path}")
    return global_step
