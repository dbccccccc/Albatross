"""
Optimized State Operations for RWKV-7 Continuous Batching.

Uses JIT compilation for high-performance state gather/scatter operations.
"""

import torch
from typing import List, Tuple


@torch.jit.script
def gather_states_jit(
    state0: torch.Tensor,
    state1: torch.Tensor,
    indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    JIT-compiled state gather operation.

    Args:
        state0: [n_layer, 2, max_batch, n_embd]
        state1: [n_layer, max_batch, n_head, head_size, head_size]
        indices: [batch_size] - slot indices to gather

    Returns:
        Tuple of gathered state tensors (contiguous)
    """
    # Use index_select for efficient gathering
    # state0: gather on dim 2
    gathered_state0 = state0.index_select(2, indices).contiguous()
    # state1: gather on dim 1
    gathered_state1 = state1.index_select(1, indices).contiguous()

    return gathered_state0, gathered_state1


@torch.jit.script
def scatter_states_jit(
    state0: torch.Tensor,
    state1: torch.Tensor,
    indices: torch.Tensor,
    new_state0: torch.Tensor,
    new_state1: torch.Tensor
) -> None:
    """
    JIT-compiled state scatter operation (in-place).

    Args:
        state0: [n_layer, 2, max_batch, n_embd] - target
        state1: [n_layer, max_batch, n_head, head_size, head_size] - target
        indices: [batch_size] - slot indices to scatter to
        new_state0: [n_layer, 2, batch_size, n_embd] - source
        new_state1: [n_layer, batch_size, n_head, head_size, head_size] - source
    """
    # Use index_copy_ for efficient in-place scatter (TorchScript compatible)
    # Scatter state0 on dim 2
    state0.index_copy_(2, indices, new_state0)
    # Scatter state1 on dim 1
    state1.index_copy_(1, indices, new_state1)


@torch.jit.script
def reset_state_slot_jit(
    state0: torch.Tensor,
    state1: torch.Tensor,
    slot_id: int
) -> None:
    """
    JIT-compiled single slot reset (in-place).

    Args:
        state0: [n_layer, 2, max_batch, n_embd]
        state1: [n_layer, max_batch, n_head, head_size, head_size]
        slot_id: Slot index to reset
    """
    state0[:, :, slot_id, :].zero_()
    state1[:, slot_id, :, :, :].zero_()


@torch.jit.script
def reset_state_slots_batch_jit(
    state0: torch.Tensor,
    state1: torch.Tensor,
    indices: torch.Tensor
) -> None:
    """
    JIT-compiled batch slot reset (in-place).

    Args:
        state0: [n_layer, 2, max_batch, n_embd]
        state1: [n_layer, max_batch, n_head, head_size, head_size]
        indices: [batch_size] - slot indices to reset
    """
    # Create zero tensors and use index_copy_ (TorchScript compatible)
    zeros0 = torch.zeros_like(state0.index_select(2, indices))
    state0.index_copy_(2, indices, zeros0)

    zeros1 = torch.zeros_like(state1.index_select(1, indices))
    state1.index_copy_(1, indices, zeros1)


class OptimizedStateOperations:
    """
    Wrapper class for optimized state operations.

    Provides both JIT-compiled and fallback implementations.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._jit_warmup_done = False

    def warmup(self, state0: torch.Tensor, state1: torch.Tensor):
        """
        Warmup JIT compilation with actual tensor shapes.
        Call this once after creating the state pool.
        """
        if self._jit_warmup_done:
            return

        # Create dummy indices for warmup
        dummy_indices = torch.tensor([0], device=self.device, dtype=torch.long)

        # Warmup gather
        _ = gather_states_jit(state0, state1, dummy_indices)

        # Warmup scatter
        gathered0, gathered1 = gather_states_jit(state0, state1, dummy_indices)
        scatter_states_jit(state0, state1, dummy_indices, gathered0, gathered1)

        # Warmup reset
        reset_state_slot_jit(state0, state1, 0)

        self._jit_warmup_done = True

    def gather(
        self,
        state0: torch.Tensor,
        state1: torch.Tensor,
        slot_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather states for specified slots.

        Args:
            state0: Full state0 tensor
            state1: Full state1 tensor
            slot_ids: List of slot IDs to gather

        Returns:
            Tuple of (gathered_state0, gathered_state1)
        """
        if not slot_ids:
            # Return empty tensors with correct shape
            n_layer = state0.shape[0]
            n_embd = state0.shape[3]
            n_head = state1.shape[2]
            head_size = state1.shape[3]

            return (
                torch.empty((n_layer, 2, 0, n_embd),
                           dtype=state0.dtype, device=state0.device),
                torch.empty((n_layer, 0, n_head, head_size, head_size),
                           dtype=state1.dtype, device=state1.device)
            )

        indices = torch.tensor(slot_ids, device=self.device, dtype=torch.long)
        return gather_states_jit(state0, state1, indices)

    def scatter(
        self,
        state0: torch.Tensor,
        state1: torch.Tensor,
        slot_ids: List[int],
        new_state0: torch.Tensor,
        new_state1: torch.Tensor
    ) -> None:
        """
        Scatter states back to specified slots.

        Args:
            state0: Full state0 tensor (modified in-place)
            state1: Full state1 tensor (modified in-place)
            slot_ids: List of slot IDs to scatter to
            new_state0: New state0 values
            new_state1: New state1 values
        """
        if not slot_ids:
            return

        indices = torch.tensor(slot_ids, device=self.device, dtype=torch.long)
        scatter_states_jit(state0, state1, indices, new_state0, new_state1)

    def reset_slot(
        self,
        state0: torch.Tensor,
        state1: torch.Tensor,
        slot_id: int
    ) -> None:
        """Reset a single slot to zero."""
        reset_state_slot_jit(state0, state1, slot_id)

    def reset_slots(
        self,
        state0: torch.Tensor,
        state1: torch.Tensor,
        slot_ids: List[int]
    ) -> None:
        """Reset multiple slots to zero."""
        if not slot_ids:
            return

        indices = torch.tensor(slot_ids, device=self.device, dtype=torch.long)
        reset_state_slots_batch_jit(state0, state1, indices)


# Global instance for convenience
_state_ops: OptimizedStateOperations = None


def get_state_ops(device: str = "cuda") -> OptimizedStateOperations:
    """Get or create the global state operations instance."""
    global _state_ops
    if _state_ops is None:
        _state_ops = OptimizedStateOperations(device)
    return _state_ops
