"""Load test the deployed API, 10 concurrent requests, p50/p95 latency.

Phase 4 step 4. Expects queueing not scaling: --max-instances=1 (D29 cost
ceiling) means concurrent requests serialize on one GPU instance.
"""
import asyncio
import statistics
import time

import httpx

URL = "https://rag-api-121020284283.asia-southeast1.run.app/chat"
N_REQUESTS = 10
QUESTION = "What is Firza's current role?"


async def one_request(client, i):
    start = time.perf_counter()
    async with client.stream(
        "POST", URL, json={"messages": [{"role": "user", "content": QUESTION, "time": "2026-09-04T00:00:00Z"}]}
    ) as resp:
        async for _ in resp.aiter_bytes():
            pass
    elapsed = time.perf_counter() - start
    print(f"  req {i}: {elapsed:.2f}s status={resp.status_code}")
    return elapsed


async def main():
    async with httpx.AsyncClient(timeout=120) as client:
        start = time.perf_counter()
        results = await asyncio.gather(*[one_request(client, i) for i in range(N_REQUESTS)])
        total = time.perf_counter() - start

    results.sort()
    p50 = statistics.median(results)
    p95 = results[int(len(results) * 0.95) - 1]
    print(f"\ntotal wall clock: {total:.2f}s")
    print(f"p50: {p50:.2f}s  p95: {p95:.2f}s  min: {results[0]:.2f}s  max: {results[-1]:.2f}s")


if __name__ == "__main__":
    # self check: script runs and produces N_REQUESTS timings, no eval framework needed
    asyncio.run(main())
