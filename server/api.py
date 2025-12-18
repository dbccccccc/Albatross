"""
FastAPI Server for RWKV-7 Inference.

Provides OpenAI-compatible REST API with SSE streaming support.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, AsyncGenerator, TYPE_CHECKING, Dict
import asyncio
import uuid
import time
import json

if TYPE_CHECKING:
    from .inference_engine import ContinuousBatchingEngine


# ============== Request/Response Models ==============

class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""
    model: str = "rwkv-7"
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    seed: Optional[int] = Field(default=None, ge=0)
    stop: Optional[List[str]] = None
    stream: bool = False
    user: Optional[str] = None


class ChatMessage(BaseModel):
    """Chat message."""
    role: str  # "system", "user", "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = "rwkv-7"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    seed: Optional[int] = Field(default=None, ge=0)
    stop: Optional[List[str]] = None
    stream: bool = False
    user: Optional[str] = None


class CompletionChoice(BaseModel):
    """Completion choice."""
    text: str
    index: int
    finish_reason: Optional[str]


class CompletionResponse(BaseModel):
    """Completion response."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: dict


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""
    index: int
    message: ChatMessage
    finish_reason: Optional[str]


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: dict


# ============== API Server ==============

class RWKVAPIServer:
    """
    FastAPI-based API server for RWKV inference.
    Provides OpenAI-compatible endpoints with SSE streaming support.
    """

    def __init__(self, engine: "ContinuousBatchingEngine"):
        self.engine = engine
        self.app = FastAPI(
            title="RWKV-7 Inference API",
            description="High-performance RWKV-7 inference with continuous batching",
            version="1.0.0"
        )
        self._setup_routes()
        self._setup_middleware()

    def _setup_middleware(self):
        """Configure CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Register API routes."""

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            status = self.engine.scheduler.get_queue_status()
            return {
                "status": "healthy",
                "queue": status,
                "timestamp": time.time()
            }

        @self.app.get("/v1/models")
        async def list_models():
            """List available models (OpenAI compatible)."""
            return {
                "object": "list",
                "data": [
                    {
                        "id": "rwkv-7",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "rwkv"
                    }
                ]
            }

        @self.app.post("/v1/completions")
        async def create_completion(request: CompletionRequest):
            """Create completion (OpenAI compatible)."""
            request_id = f"cmpl-{uuid.uuid4().hex[:24]}"

            try:
                if request.stream:
                    return StreamingResponse(
                        self._stream_completion(request, request_id),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Accel-Buffering": "no"
                        }
                    )
                else:
                    result = await self._generate_completion(request, request_id)
                    return JSONResponse(content=result)

            except RuntimeError as e:
                raise HTTPException(status_code=503, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/v1/chat/completions")
        async def create_chat_completion(request: ChatCompletionRequest):
            """Create chat completion (OpenAI compatible)."""
            request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

            # Convert chat messages to prompt
            prompt = self._format_chat_messages(request.messages)

            try:
                if request.stream:
                    return StreamingResponse(
                        self._stream_chat_completion(request, request_id, prompt),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Accel-Buffering": "no"
                        }
                    )
                else:
                    result = await self._generate_chat_completion(
                        request, request_id, prompt
                    )
                    return JSONResponse(content=result)

            except RuntimeError as e:
                raise HTTPException(status_code=503, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/v1/requests/{request_id}")
        async def cancel_request(request_id: str):
            """Cancel a pending or active request."""
            success = self.engine.scheduler.cancel_request(request_id)
            if success:
                return {"status": "cancelled", "request_id": request_id}
            else:
                raise HTTPException(status_code=404, detail="Request not found")

        @self.app.get("/v1/queue/status")
        async def queue_status():
            """Get detailed queue status."""
            return self.engine.scheduler.get_queue_status()

        @self.app.get("/v1/stats")
        async def engine_stats():
            """Get engine statistics."""
            return self.engine.get_stats()

    def _convert_logit_bias(self, logit_bias: Optional[Dict[str, float]]) -> Optional[Dict[int, float]]:
        """Convert logit_bias from string keys (OpenAI format) to int keys."""
        if logit_bias is None:
            return None
        return {int(k): v for k, v in logit_bias.items()}

    def _format_chat_messages(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to RWKV prompt format."""
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}\n\n")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}\n\n")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}\n\n")

        # Add assistant prefix for generation
        prompt_parts.append("Assistant:")
        return "".join(prompt_parts)

    async def _generate_completion(
        self,
        request: CompletionRequest,
        request_id: str
    ) -> dict:
        """Generate non-streaming completion."""
        result = await self.engine.generate(
            request_id=request_id,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            logit_bias=self._convert_logit_bias(request.logit_bias),
            seed=request.seed,
            stop_sequences=request.stop,
            stream=False
        )

        output_text = self.engine.tokenizer.decode(result['tokens'])
        prompt_tokens = len(self.engine.tokenizer.encode(request.prompt))

        return {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "text": output_text,
                "index": 0,
                "finish_reason": result['finish_reason']
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(result['tokens']),
                "total_tokens": prompt_tokens + len(result['tokens'])
            }
        }

    async def _stream_completion(
        self,
        request: CompletionRequest,
        request_id: str
    ) -> AsyncGenerator[str, None]:
        """Stream completion using Server-Sent Events."""
        stream_queue = await self.engine.generate(
            request_id=request_id,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            logit_bias=self._convert_logit_bias(request.logit_bias),
            seed=request.seed,
            stop_sequences=request.stop,
            stream=True
        )

        try:
            while True:
                text, is_final = await asyncio.wait_for(
                    stream_queue.get(),
                    timeout=60.0
                )

                if is_final:
                    # Send final chunk
                    chunk = {
                        "id": request_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "text": "",
                            "index": 0,
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                else:
                    # Send text chunk
                    chunk = {
                        "id": request_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "text": text,
                            "index": 0,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'error': 'timeout'})}\n\n"
            yield "data: [DONE]\n\n"

    async def _generate_chat_completion(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        prompt: str
    ) -> dict:
        """Generate non-streaming chat completion."""
        result = await self.engine.generate(
            request_id=request_id,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            logit_bias=self._convert_logit_bias(request.logit_bias),
            seed=request.seed,
            stop_sequences=request.stop,
            stream=False
        )

        output_text = self.engine.tokenizer.decode(result['tokens'])
        prompt_tokens = len(self.engine.tokenizer.encode(prompt))

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output_text
                },
                "finish_reason": result['finish_reason']
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(result['tokens']),
                "total_tokens": prompt_tokens + len(result['tokens'])
            }
        }

    async def _stream_chat_completion(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        prompt: str
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion using Server-Sent Events."""
        stream_queue = await self.engine.generate(
            request_id=request_id,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            logit_bias=self._convert_logit_bias(request.logit_bias),
            seed=request.seed,
            stop_sequences=request.stop,
            stream=True
        )

        try:
            while True:
                text, is_final = await asyncio.wait_for(
                    stream_queue.get(),
                    timeout=60.0
                )

                if is_final:
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                else:
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'error': 'timeout'})}\n\n"
            yield "data: [DONE]\n\n"

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API server."""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)
