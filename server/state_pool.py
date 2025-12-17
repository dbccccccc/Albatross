"""
State Pool Manager for RWKV-7 Continuous Batching.

Pre-allocates state memory for maximum batch size and provides O(1) slot allocation/deallocation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Set, Callable, Dict
from enum import Enum
from collections import deque
import threading
import logging
import torch

from .config import StatePoolConfig
from .state_operations import OptimizedStateOperations

logger = logging.getLogger(__name__)


class SlotStatus(Enum):
    """Status of a state slot."""
    FREE = 0        # Available for allocation
    PREFILL = 1     # Processing prompt (prefill phase)
    DECODE = 2      # Generating tokens (decode phase)
    PENDING = 3     # Waiting to be added to batch


@dataclass
class StateSlot:
    """Represents a single state slot in the pool."""
    slot_id: int
    status: SlotStatus = SlotStatus.FREE
    request_id: Optional[str] = None
    tokens_generated: int = 0
    max_tokens: int = 0
    output_callback: Optional[Callable] = None
    temperature: float = 1.0
    top_p: float = 1.0
    output_tokens: List[int] = field(default_factory=list)
    stop_tokens: Set[int] = field(default_factory=lambda: {0})
    prompt_tokens: List[int] = field(default_factory=list)
    prefill_pos: int = 0

    def reset(self):
        """Reset slot to initial state."""
        self.status = SlotStatus.FREE
        self.request_id = None
        self.tokens_generated = 0
        self.max_tokens = 0
        self.output_callback = None
        self.temperature = 1.0
        self.top_p = 1.0
        self.output_tokens = []
        self.stop_tokens = {0}
        self.prompt_tokens = []
        self.prefill_pos = 0


class StatePool:
    """
    Pre-allocated state pool for continuous batching.

    Key Design Decisions:
    1. Pre-allocate maximum batch size to avoid runtime allocation
    2. Use slot-based management for O(1) allocation/deallocation
    3. Maintain contiguous memory layout for CUDA kernel compatibility

    State Structure:
    - state0: [n_layer, 2, max_batch, n_embd] - Time-mixing state (FP16)
    - state1: [n_layer, max_batch, n_head, head_size, head_size] - Attention state (FP32)
    """

    def __init__(self, config: StatePoolConfig):
        self.config = config
        self.lock = threading.RLock()

        # Pre-allocate state tensors for maximum batch size
        # state0: [n_layer, 2, max_batch, n_embd] - time-mixing state
        self.state0 = torch.zeros(
            (config.n_layer, 2, config.max_batch_size, config.n_embd),
            dtype=config.dtype_state0,
            device=config.device,
            requires_grad=False
        )

        # state1: [n_layer, max_batch, n_head, head_size, head_size] - attention state
        self.state1 = torch.zeros(
            (config.n_layer, config.max_batch_size, config.n_head,
             config.head_size, config.head_size),
            dtype=config.dtype_state1,
            device=config.device,
            requires_grad=False
        )

        # Slot management
        self.slots: List[StateSlot] = [
            StateSlot(slot_id=i) for i in range(config.max_batch_size)
        ]

        # Free slot queue for O(1) allocation
        self.free_slots: deque = deque(range(config.max_batch_size))

        # Request ID to slot mapping
        self.request_to_slot: Dict[str, int] = {}

        # Optimized state operations (JIT compiled)
        self.state_ops = OptimizedStateOperations(device=config.device)
        self.state_ops.warmup(self.state0, self.state1)
        logger.info("State pool initialized with JIT-optimized operations")

    def allocate_slot(
        self,
        request_id: str,
        prompt_tokens: List[int],
        max_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        output_callback: Optional[Callable] = None,
        stop_tokens: Optional[Set[int]] = None
    ) -> Optional[int]:
        """
        Allocate a state slot for a new request.

        Args:
            request_id: Unique identifier for the request
            prompt_tokens: Tokenized prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            output_callback: Callback for streaming output
            stop_tokens: Set of token IDs that stop generation

        Returns:
            slot_id or None if pool is full
        """
        with self.lock:
            if not self.free_slots:
                return None

            slot_id = self.free_slots.popleft()
            slot = self.slots[slot_id]

            # Reset state tensors to zero for this slot
            self.state0[:, :, slot_id, :].zero_()
            self.state1[:, slot_id, :, :, :].zero_()

            # Initialize slot metadata
            slot.status = SlotStatus.PENDING
            slot.request_id = request_id
            slot.tokens_generated = 0
            slot.max_tokens = max_tokens
            slot.output_callback = output_callback
            slot.temperature = temperature
            slot.top_p = top_p
            slot.output_tokens = []
            slot.stop_tokens = stop_tokens if stop_tokens else {0}
            slot.prompt_tokens = prompt_tokens
            slot.prefill_pos = 0

            self.request_to_slot[request_id] = slot_id
            return slot_id

    def release_slot(self, slot_id: int):
        """Release a slot back to the pool."""
        with self.lock:
            slot = self.slots[slot_id]
            if slot.request_id and slot.request_id in self.request_to_slot:
                del self.request_to_slot[slot.request_id]

            slot.reset()
            self.free_slots.append(slot_id)

    def get_slot(self, slot_id: int) -> StateSlot:
        """Get slot by ID."""
        return self.slots[slot_id]

    def get_slot_by_request(self, request_id: str) -> Optional[StateSlot]:
        """Get slot by request ID."""
        with self.lock:
            slot_id = self.request_to_slot.get(request_id)
            if slot_id is not None:
                return self.slots[slot_id]
            return None

    def get_active_slots(self, status_filter: Optional[List[SlotStatus]] = None) -> List[int]:
        """
        Get list of active slot IDs, optionally filtered by status.

        Args:
            status_filter: List of statuses to include (default: PREFILL, DECODE)

        Returns:
            List of slot IDs matching the filter
        """
        with self.lock:
            if status_filter is None:
                status_filter = [SlotStatus.PREFILL, SlotStatus.DECODE]
            return [
                slot.slot_id for slot in self.slots
                if slot.status in status_filter
            ]

    def get_batch_state_view(self, slot_ids: List[int]) -> List[torch.Tensor]:
        """
        Create a contiguous state view for the given slots.

        CRITICAL: CUDA kernel requires contiguous batch indices.
        Uses JIT-compiled gather operation for performance.

        Args:
            slot_ids: List of slot IDs to include in batch

        Returns:
            [state0_batch, state1_batch] - Contiguous state tensors
        """
        state0_batch, state1_batch = self.state_ops.gather(
            self.state0, self.state1, slot_ids
        )
        return [state0_batch, state1_batch]

    def scatter_state_back(self, slot_ids: List[int], batch_state: List[torch.Tensor]):
        """
        Scatter the modified batch state back to the pool.

        Called after forward pass to update individual slot states.
        Uses JIT-compiled scatter operation for performance.

        Args:
            slot_ids: List of slot IDs that were processed
            batch_state: [state0_batch, state1_batch] from forward pass
        """
        self.state_ops.scatter(
            self.state0, self.state1, slot_ids,
            batch_state[0], batch_state[1]
        )

    def reset_slot_state(self, slot_id: int):
        """Reset state tensors for a specific slot to zero."""
        self.state0[:, :, slot_id, :].zero_()
        self.state1[:, slot_id, :, :, :].zero_()

    @property
    def available_slots(self) -> int:
        """Number of available slots."""
        return len(self.free_slots)

    @property
    def active_count(self) -> int:
        """Number of active requests."""
        return self.config.max_batch_size - len(self.free_slots)

    def get_stats(self) -> Dict:
        """Get pool statistics."""
        with self.lock:
            status_counts = {}
            for status in SlotStatus:
                status_counts[status.name] = sum(
                    1 for slot in self.slots if slot.status == status
                )

            return {
                "max_batch_size": self.config.max_batch_size,
                "available_slots": self.available_slots,
                "active_count": self.active_count,
                "status_counts": status_counts,
            }
