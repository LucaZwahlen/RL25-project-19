# Taken from https://github.com/timoklein/redo/blob/main/src/redo.py

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


@torch.inference_mode()
def _get_activation(name: str, activations: dict[str, torch.Tensor]):
    """Fetches and stores the activations of a network layer."""

    def hook(layer: nn.Linear | nn.Conv2d, input: tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        Get the activations of a layer with relu nonlinearity.
        ReLU has to be called explicitly here because the hook is attached to the conv/linear layer.
        """
        activations[name] = F.relu(output)
        if 'weight_mask' in [buffer[0] for buffer in layer.named_buffers()]:
            # only select the active neurons
            # reduce all dims except the output dim which is 0
            # check if conv layer
            if isinstance(layer, nn.Conv2d):
                mask_output_nonzero = layer.weight_mask.sum(dim=(1, 2, 3)) != 0
            elif isinstance(layer, nn.Linear):
                mask_output_nonzero = layer.weight_mask.sum(dim=1) != 0
            else:
                raise ValueError("Only Conv2d and Linear layers are supported")
            activations[name] = activations[name][:, mask_output_nonzero]
        else:
            activations[name] = activations[name]

    return hook


@torch.inference_mode()
def _get_redo_masks(activations: dict[str, torch.Tensor], tau: float) -> torch.Tensor:
    """
    Computes the ReDo mask for a given set of activations.
    The returned mask has True where neurons are dormant and False where they are active.
    """
    masks = []
    names = []
    # Remove the layers that are considered output layers like the critic, actor, etc.
    # valid_activations = [(name, activation) for name, activation in list(activations.items()) if
    #                      name not in ["critic", "actor", "q", "q1", "q2", "value", "policy"]]
    valid_activations = [
        (name, activation) for name, activation in list(activations.items()) if
        all(out_name not in name for out_name in ["critic", "actor", "q", "q1", "q2", "value", "policy"])
    ]

    # print all names
    # print([name for name, activation in valid_activations])

    for name, activation in valid_activations:
        # Taking the mean here conforms to the expectation under D in the main paper's formula
        if activation.ndim == 4:
            # Conv layer
            score = activation.abs().mean(dim=(0, 2, 3))
        else:
            # Linear layer
            score = activation.abs().mean(dim=0)

        # Divide by activation mean to make the threshold independent of the layer size
        # see https://github.com/google/dopamine/blob/ce36aab6528b26a699f5f1cefd330fdaf23a5d72/dopamine/labs/redo/weight_recyclers.py#L314
        # https://github.com/google/dopamine/issues/209
        normalized_score = score / (score.mean() + 1e-9)

        layer_mask = torch.zeros_like(normalized_score, dtype=torch.bool)
        if tau > 0.0:
            layer_mask[normalized_score <= tau] = 1
        else:
            layer_mask[torch.isclose(normalized_score, torch.zeros_like(normalized_score))] = 1
        masks.append(layer_mask)
        names.append(name)
    return masks, names


@torch.no_grad()
def run_redo(
        obs,
        model,
        optimizer: optim.Adam,
        tau: float,
        re_initialize: bool,
        use_lecun_init: bool,
) -> dict:  # tuple[nn.Module, optim.Adam, float, int]:
    """
    Checks the number of dormant neurons for a given model.
    If re_initialize is True, then the dormant neurons are re-initialized according to the scheme in
    https://arxiv.org/abs/2302.12902

    Returns the number of dormant neurons.
    """
    obs = obs[:256] if len(obs) > 256 else obs

    with torch.inference_mode():
        activations = {}
        activation_getter = partial(_get_activation, activations=activations)

        # Register hooks for all Conv2d and Linear layers to calculate activations
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                handles.append(module.register_forward_hook(activation_getter(name)))

            # Calculate activations
            if hasattr(model.forward, "_torchdynamo_orig_callable"):
                _ = model._torchdynamo_orig_callable(obs)
            else:
                _ = model(obs)

        # Masks for tau=0 logging
        zero_masks, _ = _get_redo_masks(activations, 0.0)
        total_neurons = sum([torch.numel(mask) for mask in zero_masks])
        zero_count = sum([torch.sum(mask) for mask in zero_masks])
        zero_fraction = (zero_count / total_neurons) * 100

        # Calculate the masks actually used for resetting
        masks, names = _get_redo_masks(activations, tau)

        dormant_neurons_per_layer = {name: mask.sum().item() / mask.numel() * 100 for name, mask in zip(names, masks)}
        # if the name contains _orig_mod, remove it form the name
        for name in list(dormant_neurons_per_layer.keys()):
            if '_orig_mod' in name:
                new_name = name.replace('_orig_mod.', '')
                dormant_neurons_per_layer[new_name] = dormant_neurons_per_layer.pop(name)

        dormant_count = sum([torch.sum(mask) for mask in masks])
        dormant_fraction = (dormant_count / sum([torch.numel(mask) for mask in masks])) * 100

        # Remove the hooks again
        for handle in handles:
            handle.remove()

        return {
            "model": model,
            "optimizer": optimizer,
            "zero_fraction": zero_fraction,
            "zero_count": zero_count,
            "dormant_fraction": dormant_fraction,
            "dormant_count": dormant_count,
            "dormant_neurons_per_layer": dormant_neurons_per_layer,
        }
