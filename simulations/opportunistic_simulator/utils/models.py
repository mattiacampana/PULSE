from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def fuse_model_states(
    states: Sequence[Mapping[str, torch.Tensor]],
    *,
    device: torch.device,
    weights: Sequence[float] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Fuse compatible model state dicts.

    Floating/complex tensors are averaged, optionally with normalized weights.
    Non-floating tensors (e.g. BatchNorm counters) are kept from the first state.
    """
    if not states:
        raise ValueError("At least one state is required.")

    if weights is None:
        weights_tensor = torch.full(
            (len(states),),
            1.0 / len(states),
            device=device,
        )
    else:
        if len(weights) != len(states):
            raise ValueError("weights must have the same length as states.")

        weights_tensor = torch.tensor(
            weights,
            dtype=torch.float32,
            device=device,
        )

        if torch.any(weights_tensor < 0) or weights_tensor.sum() <= 0:
            raise ValueError("weights must be non-negative and sum to a positive value.")

        weights_tensor = weights_tensor / weights_tensor.sum()

    reference_state = states[0]
    fused_state: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for name, reference_tensor in reference_state.items():
            tensors = [state[name] for state in states]

            if reference_tensor.is_floating_point() or reference_tensor.is_complex():
                stacked = torch.stack([
                    tensor.to(device=device, dtype=reference_tensor.dtype)
                    for tensor in tensors
                ])

                fused_state[name] = (
                    stacked
                    * weights_tensor.view(
                        len(states),
                        *([1] * (stacked.ndim - 1)),
                    )
                ).sum(dim=0)
            else:
                fused_state[name] = reference_tensor.to(device)

    return fused_state