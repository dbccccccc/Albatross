"""
Configuration classes for RWKV-7 inference server.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class StatePoolConfig:
    """Configuration for state pool manager."""
    max_batch_size: int = 256
    n_layer: int = 32
    n_embd: int = 4096
    head_size: int = 64
    device: str = "cuda"
    dtype_state0: torch.dtype = torch.float16
    dtype_state1: torch.dtype = torch.float32

    @property
    def n_head(self) -> int:
        return self.n_embd // self.head_size


@dataclass
class SchedulerConfig:
    """Configuration for request scheduler."""
    max_batch_size: int = 256
    max_prefill_batch: int = 32
    max_waiting_requests: int = 1000
    prefill_chunk_size: int = 512
    scheduling_interval_ms: float = 1.0
    request_timeout_seconds: float = 300.0


@dataclass
class EngineConfig:
    """Configuration for continuous batching engine."""
    max_batch_size: int = 256
    prefill_chunk_size: int = 512
    decode_micro_batch: int = 64
    use_cuda_graph: bool = True
    cuda_graph_batch_sizes: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128, 256]
    )
    engine_loop_interval_ms: float = 0.5
    default_max_tokens: int = 256
    default_temperature: float = 1.0
    default_top_p: float = 1.0


@dataclass
class ServerConfig:
    """Configuration for API server."""
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = ""
    tokenizer_path: str = "reference/rwkv_vocab_v20230424.txt"

    # Component configs
    state_pool: StatePoolConfig = field(default_factory=StatePoolConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)

    # Server settings
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    log_level: str = "info"

    @classmethod
    def from_args(cls, args) -> "ServerConfig":
        """Create config from command line arguments."""
        config = cls(
            host=args.host,
            port=args.port,
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
        )

        # Update component configs
        config.state_pool.max_batch_size = args.max_batch_size
        config.scheduler.max_batch_size = args.max_batch_size
        config.scheduler.max_prefill_batch = args.max_prefill_batch
        config.scheduler.max_waiting_requests = args.max_queue_size
        config.scheduler.prefill_chunk_size = args.prefill_chunk_size
        config.engine.max_batch_size = args.max_batch_size
        config.engine.prefill_chunk_size = args.prefill_chunk_size
        config.engine.use_cuda_graph = args.use_cuda_graph

        return config
