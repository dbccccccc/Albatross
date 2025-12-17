"""
Test script for RWKV-7 Inference Server.

Usage:
    # First start the server:
    python -m server.main --model-path /path/to/model --port 8000

    # Then run this test script:
    python scripts/test_server.py --base-url http://localhost:8000
"""

import argparse
import requests
import json
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def test_health(base_url: str) -> bool:
    """Test health endpoint."""
    print("\n[Test] Health Check...")
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Status: {data.get('status')}")
            print(f"  Queue: {data.get('queue')}")
            print("  [PASS] Health check passed")
            return True
        else:
            print(f"  [FAIL] Status code: {resp.status_code}")
            return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_models(base_url: str) -> bool:
    """Test models endpoint."""
    print("\n[Test] List Models...")
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get('data', [])
            print(f"  Available models: {[m['id'] for m in models]}")
            print("  [PASS] Models endpoint passed")
            return True
        else:
            print(f"  [FAIL] Status code: {resp.status_code}")
            return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_completion(base_url: str, prompt: str = "Hello, how are you?",
                    max_tokens: int = 50) -> bool:
    """Test completion endpoint (non-streaming)."""
    print(f"\n[Test] Completion (non-streaming)...")
    print(f"  Prompt: {prompt[:50]}...")

    try:
        start_time = time.time()
        resp = requests.post(
            f"{base_url}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.8,
            },
            timeout=60
        )
        elapsed = time.time() - start_time

        if resp.status_code == 200:
            data = resp.json()
            text = data['choices'][0]['text']
            usage = data.get('usage', {})
            print(f"  Response: {text[:100]}...")
            print(f"  Tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"  Time: {elapsed:.2f}s")
            print("  [PASS] Completion passed")
            return True
        else:
            print(f"  [FAIL] Status code: {resp.status_code}")
            print(f"  Response: {resp.text}")
            return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_completion_streaming(base_url: str, prompt: str = "Tell me a short story.",
                              max_tokens: int = 100) -> bool:
    """Test completion endpoint (streaming)."""
    print(f"\n[Test] Completion (streaming)...")
    print(f"  Prompt: {prompt[:50]}...")

    try:
        start_time = time.time()
        resp = requests.post(
            f"{base_url}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.8,
                "stream": True,
            },
            stream=True,
            timeout=60
        )

        if resp.status_code == 200:
            chunks = []
            print("  Streaming: ", end="", flush=True)
            for line in resp.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            text = chunk['choices'][0].get('text', '')
                            chunks.append(text)
                            print(text, end="", flush=True)
                        except json.JSONDecodeError:
                            pass

            elapsed = time.time() - start_time
            print()
            print(f"  Total chunks: {len(chunks)}")
            print(f"  Time: {elapsed:.2f}s")
            print("  [PASS] Streaming passed")
            return True
        else:
            print(f"  [FAIL] Status code: {resp.status_code}")
            return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_chat_completion(base_url: str) -> bool:
    """Test chat completion endpoint."""
    print("\n[Test] Chat Completion...")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]

    try:
        start_time = time.time()
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "messages": messages,
                "max_tokens": 50,
                "temperature": 0.5,
            },
            timeout=60
        )
        elapsed = time.time() - start_time

        if resp.status_code == 200:
            data = resp.json()
            content = data['choices'][0]['message']['content']
            print(f"  Response: {content[:100]}...")
            print(f"  Time: {elapsed:.2f}s")
            print("  [PASS] Chat completion passed")
            return True
        else:
            print(f"  [FAIL] Status code: {resp.status_code}")
            print(f"  Response: {resp.text}")
            return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_concurrent_requests(base_url: str, num_requests: int = 5) -> bool:
    """Test concurrent requests."""
    print(f"\n[Test] Concurrent Requests ({num_requests} requests)...")

    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about programming.",
        "What is machine learning?",
        "Tell me a joke.",
    ][:num_requests]

    def make_request(prompt):
        start = time.time()
        resp = requests.post(
            f"{base_url}/v1/completions",
            json={"prompt": prompt, "max_tokens": 30, "temperature": 0.8},
            timeout=60
        )
        elapsed = time.time() - start
        return resp.status_code == 200, elapsed

    try:
        start_time = time.time()
        results = []

        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = {executor.submit(make_request, p): p for p in prompts}
            for future in as_completed(futures):
                success, elapsed = future.result()
                results.append((success, elapsed))

        total_time = time.time() - start_time
        successes = sum(1 for s, _ in results if s)
        avg_time = sum(e for _, e in results) / len(results)

        print(f"  Successful: {successes}/{num_requests}")
        print(f"  Avg response time: {avg_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")

        if successes == num_requests:
            print("  [PASS] Concurrent requests passed")
            return True
        else:
            print("  [FAIL] Some requests failed")
            return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_queue_status(base_url: str) -> bool:
    """Test queue status endpoint."""
    print("\n[Test] Queue Status...")
    try:
        resp = requests.get(f"{base_url}/v1/queue/status", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Waiting: {data.get('waiting', 0)}")
            print(f"  Prefill: {data.get('prefill', 0)}")
            print(f"  Decode: {data.get('decode', 0)}")
            print(f"  Available slots: {data.get('available_slots', 0)}")
            print("  [PASS] Queue status passed")
            return True
        else:
            print(f"  [FAIL] Status code: {resp.status_code}")
            return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test RWKV-7 Inference Server")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000",
                        help="Base URL of the server")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip generation tests (only test health/status)")
    parser.add_argument("--concurrent", type=int, default=5,
                        help="Number of concurrent requests to test")
    args = parser.parse_args()

    print("=" * 60)
    print("RWKV-7 Inference Server Test Suite")
    print(f"Target: {args.base_url}")
    print("=" * 60)

    results = []

    # Basic tests
    results.append(("Health Check", test_health(args.base_url)))
    results.append(("Models List", test_models(args.base_url)))
    results.append(("Queue Status", test_queue_status(args.base_url)))

    # Generation tests
    if not args.skip_generation:
        results.append(("Completion", test_completion(args.base_url)))
        results.append(("Streaming", test_completion_streaming(args.base_url)))
        results.append(("Chat", test_chat_completion(args.base_url)))
        results.append(("Concurrent", test_concurrent_requests(args.base_url, args.concurrent)))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: [{status}]")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
