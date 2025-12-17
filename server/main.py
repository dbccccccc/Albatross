"""
RWKV-7 Inference Server Entry Point.

Usage:
    python -m server.main --model-path /path/to/model --port 8000
"""

import argparse
import sys
import types
import os
import signal
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RWKV-7 High-Concurrency Inference Server"
    )

    # Model settings
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to model weights (without .pth extension)"
    )
    parser.add_argument(
        "--tokenizer-path", type=str,
        default="reference/rwkv_vocab_v20230424.txt",
        help="Path to tokenizer vocabulary file"
    )

    # Server settings
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Server port (default: 8000)"
    )

    # Batch settings
    parser.add_argument(
        "--max-batch-size", type=int, default=256,
        help="Maximum batch size for inference (default: 256)"
    )
    parser.add_argument(
        "--max-prefill-batch", type=int, default=32,
        help="Maximum requests in prefill per iteration (default: 32)"
    )
    parser.add_argument(
        "--max-queue-size", type=int, default=1000,
        help="Maximum waiting queue size (default: 1000)"
    )
    parser.add_argument(
        "--prefill-chunk-size", type=int, default=512,
        help="Chunk size for long prompt prefill (default: 512)"
    )

    # Performance settings
    parser.add_argument(
        "--use-cuda-graph", action="store_true",
        help="Enable CUDA Graph for decode phase (experimental)"
    )

    return parser.parse_args()


def create_server(args):
    """Create and configure the inference server."""
    import torch

    from server.config import StatePoolConfig, SchedulerConfig, EngineConfig
    from server.state_pool import StatePool
    from server.scheduler import RequestScheduler
    from server.inference_engine import ContinuousBatchingEngine
    from server.api import RWKVAPIServer

    # Import model and tokenizer
    from reference.rwkv7 import RWKV_x070
    from reference.utils import TRIE_TOKENIZER

    print(f"Loading model from {args.model_path}...")

    # Create model args
    model_args = types.SimpleNamespace()
    model_args.vocab_size = 65536
    model_args.head_size = 64
    model_args.MODEL_NAME = args.model_path

    # Load model
    model = RWKV_x070(model_args)

    # Get model dimensions from loaded model
    n_layer = model.args.n_layer
    n_embd = model.args.n_embd
    head_size = model.args.head_size

    print(f"Model loaded: {n_layer} layers, {n_embd} embedding dim")

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = TRIE_TOKENIZER(args.tokenizer_path)

    # Create state pool
    print(f"Creating state pool (max_batch_size={args.max_batch_size})...")
    pool_config = StatePoolConfig(
        max_batch_size=args.max_batch_size,
        n_layer=n_layer,
        n_embd=n_embd,
        head_size=head_size
    )
    state_pool = StatePool(pool_config)

    # Create scheduler
    scheduler_config = SchedulerConfig(
        max_batch_size=args.max_batch_size,
        max_prefill_batch=args.max_prefill_batch,
        max_waiting_requests=args.max_queue_size,
        prefill_chunk_size=args.prefill_chunk_size
    )
    scheduler = RequestScheduler(scheduler_config, state_pool)

    # Create engine
    engine_config = EngineConfig(
        max_batch_size=args.max_batch_size,
        prefill_chunk_size=args.prefill_chunk_size,
        use_cuda_graph=args.use_cuda_graph,
    )
    engine = ContinuousBatchingEngine(
        model=model,
        tokenizer=tokenizer,
        state_pool=state_pool,
        scheduler=scheduler,
        config=engine_config
    )

    # Create API server
    api_server = RWKVAPIServer(engine)

    return api_server, engine


def main():
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("RWKV-7 High-Concurrency Inference Server")
    logger.info("=" * 60)

    # Create server
    api_server, engine = create_server(args)

    # Setup signal handlers for graceful shutdown
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            logger.warning("Force shutdown requested")
            sys.exit(1)
        shutdown_requested = True
        logger.info("Shutdown signal received, stopping gracefully...")
        engine.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start engine
    logger.info("Starting inference engine...")
    engine.start()

    # Run API server
    logger.info(f"Starting API server on http://{args.host}:{args.port}")
    logger.info("Endpoints:")
    logger.info(f"  - Health: http://{args.host}:{args.port}/health")
    logger.info(f"  - Models: http://{args.host}:{args.port}/v1/models")
    logger.info(f"  - Completions: http://{args.host}:{args.port}/v1/completions")
    logger.info(f"  - Chat: http://{args.host}:{args.port}/v1/chat/completions")
    logger.info("=" * 60)

    try:
        api_server.run(host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
    finally:
        logger.info("Stopping inference engine...")
        engine.stop()
        logger.info("Server stopped.")


if __name__ == "__main__":
    main()
