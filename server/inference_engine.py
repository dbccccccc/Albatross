"""
Continuous Batching Inference Engine for RWKV-7.

Implements the main inference loop with prefill/decode separation.
"""

import torch
import torch.nn.functional as F
import asyncio
import threading
import time
import logging
from typing import List, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass
from concurrent.futures import Future
from collections import Counter

from .config import EngineConfig
from .state_pool import StatePool, SlotStatus
from .scheduler import RequestScheduler, InferenceRequest
from .batch_optimizer import BatchOptimizer, BatchOptimizerConfig

if TYPE_CHECKING:
    from reference.rwkv7 import RWKV_x070
    from reference.utils import TRIE_TOKENIZER

# Setup logging
logger = logging.getLogger(__name__)


class ContinuousBatchingEngine:
    """
    Main inference engine implementing continuous batching.

    Key Features:
    1. Asynchronous request handling
    2. Continuous batching with dynamic batch composition
    3. Chunked prefill for long prompts
    4. Streaming output support
    """

    def __init__(
        self,
        model: "RWKV_x070",
        tokenizer: "TRIE_TOKENIZER",
        state_pool: StatePool,
        scheduler: RequestScheduler,
        config: EngineConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.state_pool = state_pool
        self.scheduler = scheduler
        self.config = config

        self.running = False
        self.engine_thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Sync completion callbacks for generate_sync
        self._sync_completions: Dict[str, callable] = {}

        # Batch optimizer for dynamic batch size and memory management
        optimizer_config = BatchOptimizerConfig(
            max_batch_size=config.max_batch_size,
            default_prefill_chunk_size=config.prefill_chunk_size,
        )
        self.batch_optimizer = BatchOptimizer(optimizer_config)

        # Performance stats
        self.stats = {
            'total_tokens_generated': 0,
            'total_prefill_tokens': 0,
            'iterations': 0,
            'start_time': time.time(),
        }

    def start(self):
        """Start the inference engine."""
        if self.running:
            return
        self.running = True
        self.engine_thread = threading.Thread(target=self._engine_loop, daemon=True)
        self.engine_thread.start()

    def stop(self):
        """Stop the inference engine."""
        self.running = False
        if self.engine_thread:
            self.engine_thread.join(timeout=5.0)

    def _engine_loop(self):
        """Main engine loop - runs in separate thread."""
        # Create event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        logger.info("Inference engine started")

        while self.running:
            try:
                self._process_batch()
                self.stats['iterations'] += 1
            except Exception as e:
                logger.error(f"Engine error: {e}", exc_info=True)

            # Small sleep to prevent busy-waiting when idle
            time.sleep(self.config.engine_loop_interval_ms / 1000)

        logger.info("Inference engine stopped")

    def _process_batch(self):
        """Process one batch iteration."""
        # Get scheduled batch
        batch = self.scheduler.schedule_batch()

        prefill_requests = batch['prefill']
        decode_requests = batch['decode']

        # Process prefill requests (chunked)
        if prefill_requests:
            self._process_prefill(prefill_requests)

        # Process decode requests
        if decode_requests:
            self._process_decode(decode_requests)

    def _process_prefill(self, requests: List[InferenceRequest]):
        """
        Process prefill phase for requests.
        Uses dynamic chunking for long prompts based on memory pressure.
        """
        # Group requests by remaining prefill length for efficiency
        length_groups: Dict[int, List[tuple]] = {}

        for req in requests:
            slot = self.state_pool.get_slot(req.slot_id)
            remaining = slot.prompt_tokens[slot.prefill_pos:]

            # Get optimal chunk size based on memory pressure
            chunk_size = self.batch_optimizer.get_optimal_prefill_chunk_size(len(remaining))
            chunk = remaining[:chunk_size]
            chunk_len = len(chunk)

            if chunk_len not in length_groups:
                length_groups[chunk_len] = []
            length_groups[chunk_len].append((req, slot, chunk))

        # Process each length group
        for length, group in length_groups.items():
            if length == 0:
                continue

            slot_ids = [slot.slot_id for _, slot, _ in group]
            token_chunks = [chunk for _, _, chunk in group]

            # Get batch state view
            batch_state = self.state_pool.get_batch_state_view(slot_ids)

            # Forward pass
            with torch.no_grad():
                outputs = self.model.forward_batch(token_chunks, batch_state, full_output=False)

            # Scatter state back
            self.state_pool.scatter_state_back(slot_ids, batch_state)

            # Update prefill positions and check completion
            for i, (req, slot, chunk) in enumerate(group):
                slot.prefill_pos += len(chunk)
                self.stats['total_prefill_tokens'] += len(chunk)

                # If prefill complete, sample first token
                if slot.prefill_pos >= len(slot.prompt_tokens):
                    logits = outputs[i]
                    token = self._sample_token(
                        logits,
                        slot.temperature,
                        slot.top_p,
                        frequency_penalty=slot.frequency_penalty,
                        presence_penalty=slot.presence_penalty,
                        logit_bias=slot.logit_bias,
                        output_tokens=slot.output_tokens,
                        generator=slot.generator
                    )

                    slot.output_tokens.append(token)
                    slot.tokens_generated = 1

                    # Stream callback
                    if slot.output_callback:
                        try:
                            decoded = self.tokenizer.decode([token])
                            slot.output_callback(decoded, is_final=False)
                        except Exception:
                            pass

    def _process_decode(self, requests: List[InferenceRequest]):
        """
        Process decode phase for requests.
        All requests generate one token per iteration.
        """
        if not requests:
            return

        # Filter out requests that just finished prefill (already got first token)
        active_requests = []
        for req in requests:
            slot = self.state_pool.get_slot(req.slot_id)
            # Only process if we have at least one output token
            if slot.output_tokens:
                active_requests.append(req)

        if not active_requests:
            return

        slot_ids = [req.slot_id for req in active_requests]

        # Get last generated token for each request
        input_tokens = []
        for req in active_requests:
            slot = self.state_pool.get_slot(req.slot_id)
            input_tokens.append([slot.output_tokens[-1]])

        # Get batch state view
        batch_state = self.state_pool.get_batch_state_view(slot_ids)

        # Forward pass
        with torch.no_grad():
            outputs = self.model.forward_batch(input_tokens, batch_state, full_output=False)

        # Scatter state back
        self.state_pool.scatter_state_back(slot_ids, batch_state)

        # Sample tokens and handle completions
        completed_requests = []

        for i, req in enumerate(active_requests):
            slot = self.state_pool.get_slot(req.slot_id)
            logits = outputs[i]

            # Sample next token
            token = self._sample_token(
                logits,
                slot.temperature,
                slot.top_p,
                frequency_penalty=slot.frequency_penalty,
                presence_penalty=slot.presence_penalty,
                logit_bias=slot.logit_bias,
                output_tokens=slot.output_tokens,
                generator=slot.generator
            )
            slot.output_tokens.append(token)
            slot.tokens_generated += 1
            self.stats['total_tokens_generated'] += 1

            # Stream callback
            if slot.output_callback:
                try:
                    decoded = self.tokenizer.decode([token])
                    slot.output_callback(decoded, is_final=False)
                except Exception:
                    pass

            # Check completion conditions
            is_complete = False
            finish_reason = None

            if token in slot.stop_tokens:
                is_complete = True
                finish_reason = "stop"
            elif slot.tokens_generated >= slot.max_tokens:
                is_complete = True
                finish_reason = "length"

            if is_complete:
                completed_requests.append((req, finish_reason))
                if slot.output_callback:
                    try:
                        slot.output_callback("", is_final=True)
                    except Exception:
                        pass

        # Mark completed requests
        for req, reason in completed_requests:
            slot = self.state_pool.get_slot(req.slot_id)
            output_tokens = list(slot.output_tokens)  # Copy before slot is released

            # Call sync completion callback if exists
            sync_callback = self._sync_completions.get(req.request_id)
            if sync_callback:
                try:
                    sync_callback(output_tokens, reason)
                except Exception as e:
                    logger.error(f"Sync callback error for {req.request_id}: {e}")

            self.scheduler.mark_request_complete(
                req.request_id,
                output_tokens,
                reason
            )

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: Optional[Dict[int, float]] = None,
        output_tokens: Optional[List[int]] = None,
        generator: Optional[torch.Generator] = None
    ) -> int:
        """
        Sample a token from logits with temperature, top-p, and penalties.

        Args:
            logits: Logits tensor [vocab_size]
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter (0 = disabled)
            frequency_penalty: Penalty based on token frequency (reduces repetition)
            presence_penalty: Penalty for token presence (increases diversity)
            logit_bias: Dict mapping token IDs to bias values
            output_tokens: List of previously generated tokens
            generator: Optional torch.Generator for reproducible sampling

        Returns:
            Sampled token ID
        """
        with torch.no_grad():
            # Make a copy to avoid modifying original logits
            logits = logits.float().clone()

            # 1. Apply logit_bias
            if logit_bias:
                for token_id, bias in logit_bias.items():
                    if 0 <= token_id < logits.size(-1):
                        logits[token_id] += bias

            # 2. Apply frequency and presence penalties
            if output_tokens and (frequency_penalty != 0 or presence_penalty != 0):
                token_counts = Counter(output_tokens)
                for token_id, count in token_counts.items():
                    if 0 <= token_id < logits.size(-1):
                        if frequency_penalty != 0:
                            logits[token_id] -= frequency_penalty * count
                        if presence_penalty != 0:
                            logits[token_id] -= presence_penalty

            # Greedy decoding (after penalties applied)
            if temperature == 0 or temperature < 1e-6:
                return torch.argmax(logits).item()

            # 3. Apply temperature
            logits = logits / temperature

            # 4. Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # 5. Top-k filtering
            if top_k > 0:
                top_k = min(top_k, probs.size(-1))
                top_k_probs, top_k_indices = torch.topk(probs, top_k)
                probs = torch.zeros_like(probs)
                probs.scatter_(-1, top_k_indices, top_k_probs)

            # 6. Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Find cutoff
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                probs[indices_to_remove] = 0

            # 7. Renormalize
            probs = probs / (probs.sum() + 1e-8)

            # 8. Sample (use generator for reproducible sampling if provided)
            if generator is not None:
                return torch.multinomial(probs, num_samples=1, generator=generator).item()
            else:
                return torch.multinomial(probs, num_samples=1).item()

    async def generate(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: Optional[Dict[int, float]] = None,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False
    ):
        """
        High-level generation API.

        Args:
            request_id: Unique request identifier
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            frequency_penalty: Penalty for token frequency (reduces repetition)
            presence_penalty: Penalty for token presence (increases diversity)
            logit_bias: Dict mapping token IDs to bias values
            seed: Random seed for reproducible sampling
            stop_sequences: List of stop sequences
            stream: Whether to stream output

        Returns:
            If stream=False: Dict with tokens and finish_reason
            If stream=True: AsyncGenerator yielding (text, is_final) tuples
        """
        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(prompt)

        # Get current event loop
        loop = asyncio.get_event_loop()
        completion_future = loop.create_future()

        # Encode stop sequences to token IDs
        stop_tokens = {0}  # Always include EOS
        if stop_sequences:
            for seq in stop_sequences:
                seq_tokens = self.tokenizer.encode(seq)
                if len(seq_tokens) == 1:
                    stop_tokens.add(seq_tokens[0])

        # Stream queue for streaming mode
        stream_queue = asyncio.Queue() if stream else None

        def stream_callback(text: str, is_final: bool):
            if stream_queue:
                asyncio.run_coroutine_threadsafe(
                    stream_queue.put((text, is_final)),
                    loop
                )

        # Create request
        request = InferenceRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            seed=seed,
            stop_tokens=stop_tokens,
            stream_callback=stream_callback if stream else None,
            completion_future=completion_future
        )

        # Submit to scheduler
        accepted = await self.scheduler.submit_request(request)
        if not accepted:
            raise RuntimeError("Request queue full")

        if stream:
            return stream_queue
        else:
            return await completion_future

    def generate_sync(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: Optional[Dict[int, float]] = None,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        timeout: float = 300.0,
    ) -> Dict:
        """
        Synchronous generation API (blocking).

        Args:
            request_id: Unique request identifier
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            frequency_penalty: Penalty for token frequency (reduces repetition)
            presence_penalty: Penalty for token presence (increases diversity)
            logit_bias: Dict mapping token IDs to bias values
            seed: Random seed for reproducible sampling
            stop_sequences: List of stop sequences
            timeout: Maximum time to wait (seconds)

        Returns:
            Dict with 'tokens', 'text', and 'finish_reason'
        """
        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(prompt)

        # Encode stop sequences
        stop_tokens = {0}
        if stop_sequences:
            for seq in stop_sequences:
                seq_tokens = self.tokenizer.encode(seq)
                if len(seq_tokens) == 1:
                    stop_tokens.add(seq_tokens[0])

        # Use threading Event for completion notification
        completion_event = threading.Event()
        result_holder = {'tokens': [], 'finish_reason': 'unknown'}

        def on_complete(tokens: List[int], finish_reason: str):
            result_holder['tokens'] = tokens
            result_holder['finish_reason'] = finish_reason
            completion_event.set()

        # Store completion callback
        self._sync_completions[request_id] = on_complete

        # Create request
        request = InferenceRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            seed=seed,
            stop_tokens=stop_tokens,
        )

        # Submit
        accepted = self.scheduler.submit_request_sync(request)
        if not accepted:
            del self._sync_completions[request_id]
            raise RuntimeError("Request queue full")

        # Wait for completion with timeout
        completed = completion_event.wait(timeout=timeout)

        # Cleanup
        self._sync_completions.pop(request_id, None)

        if not completed:
            # Timeout - cancel the request
            self.scheduler.cancel_request(request_id)
            raise TimeoutError(f"Request {request_id} timed out after {timeout}s")

        # Decode tokens to text
        result_holder['text'] = self.tokenizer.decode(result_holder['tokens'])
        return result_holder

    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            **self.stats,
            'queue_status': self.scheduler.get_queue_status(),
            'pool_stats': self.state_pool.get_stats(),
            'batch_optimizer': self.batch_optimizer.get_stats(),
        }
