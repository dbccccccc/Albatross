"""
Request Scheduler for RWKV-7 Continuous Batching.

Manages request queue and batch assembly using FIFO scheduling.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Set
from collections import deque
import asyncio
import time
import threading

from .config import SchedulerConfig
from .state_pool import StatePool, SlotStatus


@dataclass
class InferenceRequest:
    """Represents a single inference request."""
    request_id: str
    prompt_tokens: List[int]
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    stop_tokens: Set[int] = field(default_factory=lambda: {0})
    stream_callback: Optional[Callable] = None
    completion_future: Optional[asyncio.Future] = None

    # Runtime state (set by scheduler)
    arrival_time: float = 0.0
    slot_id: Optional[int] = None
    phase: str = "pending"  # pending, prefill, decode, done


class RequestScheduler:
    """
    Manages request queue and batch assembly for continuous batching.

    Scheduling Strategy (FIFO):
    1. All requests are treated equally (no priority)
    2. Decode requests continue until completion
    3. New prefill requests are added in FIFO order
    4. Dynamic batch composition based on available slots
    """

    def __init__(self, config: SchedulerConfig, state_pool: StatePool):
        self.config = config
        self.state_pool = state_pool
        self.lock = threading.RLock()

        # FIFO queue for waiting requests
        self.waiting_queue: deque = deque()

        # Active requests by phase
        self.prefill_requests: Dict[str, InferenceRequest] = {}
        self.decode_requests: Dict[str, InferenceRequest] = {}

        # Request lookup
        self.all_requests: Dict[str, InferenceRequest] = {}

        # Statistics
        self.stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'rejected_requests': 0,
            'cancelled_requests': 0,
        }

    async def submit_request(self, request: InferenceRequest) -> bool:
        """
        Submit a new request to the scheduler.

        Args:
            request: The inference request to submit

        Returns:
            True if accepted, False if rejected (queue full)
        """
        with self.lock:
            if len(self.waiting_queue) >= self.config.max_waiting_requests:
                self.stats['rejected_requests'] += 1
                return False

            request.arrival_time = time.time()
            request.phase = "pending"
            self.waiting_queue.append(request)
            self.all_requests[request.request_id] = request
            self.stats['total_requests'] += 1
            return True

    def submit_request_sync(self, request: InferenceRequest) -> bool:
        """Synchronous version of submit_request."""
        with self.lock:
            if len(self.waiting_queue) >= self.config.max_waiting_requests:
                self.stats['rejected_requests'] += 1
                return False

            request.arrival_time = time.time()
            request.phase = "pending"
            self.waiting_queue.append(request)
            self.all_requests[request.request_id] = request
            self.stats['total_requests'] += 1
            return True

    def schedule_batch(self) -> Dict[str, List[InferenceRequest]]:
        """
        Schedule the next batch of requests.

        Returns:
            Dict with 'prefill' and 'decode' request lists
        """
        with self.lock:
            batch = {
                'prefill': [],
                'decode': [],
            }

            # 1. Collect all decode requests (they continue)
            batch['decode'] = list(self.decode_requests.values())

            # 2. Check prefill requests - move completed ones to decode
            completed_prefill = []
            for req_id, req in list(self.prefill_requests.items()):
                slot = self.state_pool.get_slot(req.slot_id)
                if slot.prefill_pos >= len(slot.prompt_tokens):
                    # Prefill complete, move to decode
                    req.phase = 'decode'
                    slot.status = SlotStatus.DECODE
                    self.decode_requests[req_id] = req
                    completed_prefill.append(req_id)
                    batch['decode'].append(req)
                else:
                    # Continue prefill
                    batch['prefill'].append(req)

            for req_id in completed_prefill:
                del self.prefill_requests[req_id]

            # 3. Add new requests from waiting queue (FIFO)
            available_slots = self.state_pool.available_slots
            prefill_budget = min(
                self.config.max_prefill_batch - len(batch['prefill']),
                available_slots
            )

            while self.waiting_queue and prefill_budget > 0:
                request = self.waiting_queue.popleft()

                # Allocate slot
                slot_id = self.state_pool.allocate_slot(
                    request_id=request.request_id,
                    prompt_tokens=request.prompt_tokens,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    output_callback=request.stream_callback,
                    stop_tokens=request.stop_tokens
                )

                if slot_id is None:
                    # No slots available, put back at front
                    self.waiting_queue.appendleft(request)
                    break

                request.slot_id = slot_id
                request.phase = 'prefill'

                slot = self.state_pool.get_slot(slot_id)
                slot.status = SlotStatus.PREFILL

                self.prefill_requests[request.request_id] = request
                batch['prefill'].append(request)
                prefill_budget -= 1

            return batch

    def mark_request_complete(
        self,
        request_id: str,
        output_tokens: List[int],
        finish_reason: str = "stop"
    ):
        """
        Mark a request as complete and clean up.

        Args:
            request_id: ID of the completed request
            output_tokens: Generated tokens
            finish_reason: Reason for completion (stop, length, cancelled)
        """
        with self.lock:
            request = self.all_requests.get(request_id)
            if not request:
                return

            request.phase = 'done'

            # Release slot
            if request.slot_id is not None:
                self.state_pool.release_slot(request.slot_id)

            # Remove from active sets
            self.prefill_requests.pop(request_id, None)
            self.decode_requests.pop(request_id, None)

            # Complete the future
            if request.completion_future and not request.completion_future.done():
                try:
                    request.completion_future.set_result({
                        'tokens': output_tokens,
                        'finish_reason': finish_reason
                    })
                except Exception:
                    pass

            self.stats['completed_requests'] += 1
            del self.all_requests[request_id]

    def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a pending or active request.

        Args:
            request_id: ID of the request to cancel

        Returns:
            True if cancelled, False if not found
        """
        with self.lock:
            request = self.all_requests.get(request_id)
            if not request:
                return False

            if request.phase == 'pending':
                # Remove from waiting queue
                try:
                    self.waiting_queue.remove(request)
                except ValueError:
                    pass
                del self.all_requests[request_id]
            else:
                # Mark as complete with cancelled reason
                slot = self.state_pool.get_slot(request.slot_id) if request.slot_id else None
                output_tokens = slot.output_tokens if slot else []
                self.mark_request_complete(request_id, output_tokens, 'cancelled')

            self.stats['cancelled_requests'] += 1
            return True

    def get_request(self, request_id: str) -> Optional[InferenceRequest]:
        """Get request by ID."""
        with self.lock:
            return self.all_requests.get(request_id)

    def get_queue_status(self) -> Dict:
        """Get current queue status for monitoring."""
        with self.lock:
            return {
                'waiting': len(self.waiting_queue),
                'prefill': len(self.prefill_requests),
                'decode': len(self.decode_requests),
                'total_active': len(self.prefill_requests) + len(self.decode_requests),
                'available_slots': self.state_pool.available_slots,
                'stats': self.stats.copy()
            }

    def get_active_request_ids(self) -> List[str]:
        """Get list of all active request IDs."""
        with self.lock:
            return list(self.prefill_requests.keys()) + list(self.decode_requests.keys())
