"""
Batch Optimizer for RWKV-7 Continuous Batching.

Provides dynamic batch size adjustment and GPU memory monitoring.
"""

import torch
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GPUMemoryStats:
    """GPU memory statistics."""
    total_mb: float
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    utilization: float  # 0.0 - 1.0


@dataclass
class BatchOptimizerConfig:
    """Configuration for batch optimizer."""
    # Memory thresholds
    memory_high_watermark: float = 0.85  # Start reducing batch size
    memory_low_watermark: float = 0.70   # Can increase batch size
    memory_critical: float = 0.95        # Emergency reduction

    # Batch size limits
    min_batch_size: int = 1
    max_batch_size: int = 256

    # Prefill chunking
    default_prefill_chunk_size: int = 512
    min_prefill_chunk_size: int = 64
    max_prefill_chunk_size: int = 2048

    # Adjustment parameters
    batch_increase_step: int = 8
    batch_decrease_step: int = 16
    adjustment_interval_iterations: int = 10


class GPUMemoryMonitor:
    """
    Monitors GPU memory usage and provides OOM prevention.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._last_stats: Optional[GPUMemoryStats] = None

    def get_stats(self) -> GPUMemoryStats:
        """Get current GPU memory statistics."""
        if not torch.cuda.is_available():
            return GPUMemoryStats(
                total_mb=0, allocated_mb=0, reserved_mb=0,
                free_mb=0, utilization=0.0
            )

        total = torch.cuda.get_device_properties(self.device).total_memory
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        free = total - reserved

        stats = GPUMemoryStats(
            total_mb=total / (1024 * 1024),
            allocated_mb=allocated / (1024 * 1024),
            reserved_mb=reserved / (1024 * 1024),
            free_mb=free / (1024 * 1024),
            utilization=reserved / total if total > 0 else 0.0
        )

        self._last_stats = stats
        return stats

    def is_memory_critical(self, threshold: float = 0.95) -> bool:
        """Check if memory usage is critically high."""
        stats = self.get_stats()
        return stats.utilization >= threshold

    def is_memory_high(self, threshold: float = 0.85) -> bool:
        """Check if memory usage is high."""
        stats = self.get_stats()
        return stats.utilization >= threshold

    def is_memory_low(self, threshold: float = 0.70) -> bool:
        """Check if memory usage is low enough to increase batch."""
        stats = self.get_stats()
        return stats.utilization <= threshold

    def try_clear_cache(self):
        """Try to clear CUDA cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")


class BatchOptimizer:
    """
    Optimizes batch size and prefill chunk size based on GPU memory.

    Features:
    1. Dynamic batch size adjustment based on memory pressure
    2. Adaptive prefill chunk sizing for long prompts
    3. OOM prevention through proactive memory monitoring
    """

    def __init__(self, config: BatchOptimizerConfig):
        self.config = config
        self.memory_monitor = GPUMemoryMonitor()

        # Current settings
        self.current_max_batch_size = config.max_batch_size
        self.current_prefill_chunk_size = config.default_prefill_chunk_size

        # Tracking
        self._iteration_count = 0
        self._oom_count = 0
        self._last_adjustment_iteration = 0

    def get_optimal_batch_size(self, requested_batch_size: int) -> int:
        """
        Get optimal batch size considering memory constraints.

        Args:
            requested_batch_size: Desired batch size

        Returns:
            Adjusted batch size
        """
        # Check memory status
        if self.memory_monitor.is_memory_critical(self.config.memory_critical):
            # Emergency: reduce significantly
            self.current_max_batch_size = max(
                self.config.min_batch_size,
                self.current_max_batch_size // 2
            )
            logger.warning(f"Critical memory! Reducing max batch to {self.current_max_batch_size}")
            self.memory_monitor.try_clear_cache()

        elif self.memory_monitor.is_memory_high(self.config.memory_high_watermark):
            # High memory: reduce gradually
            if self._should_adjust():
                self.current_max_batch_size = max(
                    self.config.min_batch_size,
                    self.current_max_batch_size - self.config.batch_decrease_step
                )
                logger.info(f"High memory, reducing max batch to {self.current_max_batch_size}")

        elif self.memory_monitor.is_memory_low(self.config.memory_low_watermark):
            # Low memory: can increase
            if self._should_adjust():
                self.current_max_batch_size = min(
                    self.config.max_batch_size,
                    self.current_max_batch_size + self.config.batch_increase_step
                )
                logger.debug(f"Low memory, increasing max batch to {self.current_max_batch_size}")

        self._iteration_count += 1

        return min(requested_batch_size, self.current_max_batch_size)

    def get_optimal_prefill_chunk_size(self, prompt_length: int) -> int:
        """
        Get optimal prefill chunk size for a given prompt length.

        Args:
            prompt_length: Length of the prompt in tokens

        Returns:
            Optimal chunk size
        """
        # For short prompts, process in one go
        if prompt_length <= self.config.min_prefill_chunk_size:
            return prompt_length

        # Check memory and adjust chunk size
        if self.memory_monitor.is_memory_high(self.config.memory_high_watermark):
            # Reduce chunk size under memory pressure
            chunk_size = max(
                self.config.min_prefill_chunk_size,
                self.current_prefill_chunk_size // 2
            )
        else:
            chunk_size = self.current_prefill_chunk_size

        # Don't exceed prompt length
        return min(chunk_size, prompt_length)

    def report_oom(self):
        """Report an OOM event for adaptive adjustment."""
        self._oom_count += 1
        logger.error(f"OOM reported! Total OOM count: {self._oom_count}")

        # Aggressive reduction after OOM
        self.current_max_batch_size = max(
            self.config.min_batch_size,
            self.current_max_batch_size // 2
        )
        self.current_prefill_chunk_size = max(
            self.config.min_prefill_chunk_size,
            self.current_prefill_chunk_size // 2
        )

        self.memory_monitor.try_clear_cache()

    def _should_adjust(self) -> bool:
        """Check if enough iterations have passed for adjustment."""
        if self._iteration_count - self._last_adjustment_iteration >= \
           self.config.adjustment_interval_iterations:
            self._last_adjustment_iteration = self._iteration_count
            return True
        return False

    def get_stats(self) -> Dict:
        """Get optimizer statistics."""
        memory_stats = self.memory_monitor.get_stats()
        return {
            "current_max_batch_size": self.current_max_batch_size,
            "current_prefill_chunk_size": self.current_prefill_chunk_size,
            "iteration_count": self._iteration_count,
            "oom_count": self._oom_count,
            "memory": {
                "total_mb": memory_stats.total_mb,
                "allocated_mb": memory_stats.allocated_mb,
                "reserved_mb": memory_stats.reserved_mb,
                "free_mb": memory_stats.free_mb,
                "utilization": memory_stats.utilization,
            }
        }


def compute_optimal_max_batch_size(
    model_params_billions: float,
    state_size_per_slot_mb: float,
    gpu_memory_gb: float,
    safety_margin: float = 0.80
) -> int:
    """
    Compute optimal maximum batch size based on available GPU memory.

    Args:
        model_params_billions: Model size in billions of parameters
        state_size_per_slot_mb: State memory per slot in MB
        gpu_memory_gb: Total GPU memory in GB
        safety_margin: Safety margin (0.0 - 1.0)

    Returns:
        Recommended maximum batch size
    """
    # Estimate model memory (rough: 2 bytes per param for fp16)
    model_memory_mb = model_params_billions * 1000 * 2

    # Available memory for states
    available_mb = gpu_memory_gb * 1024 * safety_margin - model_memory_mb

    if available_mb <= 0:
        logger.warning("Not enough memory for batching, using minimum batch size")
        return 1

    # Calculate max batch size
    max_batch = int(available_mb / state_size_per_slot_mb)

    # Round down to power of 2 for efficiency
    if max_batch >= 256:
        return 256
    elif max_batch >= 128:
        return 128
    elif max_batch >= 64:
        return 64
    elif max_batch >= 32:
        return 32
    elif max_batch >= 16:
        return 16
    elif max_batch >= 8:
        return 8
    else:
        return max(1, max_batch)


def estimate_state_size_per_slot(
    n_layer: int,
    n_embd: int,
    head_size: int = 64
) -> float:
    """
    Estimate state memory per slot in MB.

    Args:
        n_layer: Number of layers
        n_embd: Embedding dimension
        head_size: Head size (default 64)

    Returns:
        Estimated state size in MB
    """
    n_head = n_embd // head_size

    # state0: [n_layer, 2, 1, n_embd] in fp16 (2 bytes)
    state0_size = n_layer * 2 * n_embd * 2

    # state1: [n_layer, 1, n_head, head_size, head_size] in fp32 (4 bytes)
    state1_size = n_layer * n_head * head_size * head_size * 4

    total_bytes = state0_size + state1_size
    return total_bytes / (1024 * 1024)
