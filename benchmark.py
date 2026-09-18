"""Live latency benchmark against a running GridWise service.

Usage:
    python benchmark.py [cases.json] [base_url] [repeats]

Defaults to the local participant sample pack and http://127.0.0.1:8000.
Reports median, p95, and max wall-clock latency plus failure count.
"""

import json
import statistics
import sys
import time
import urllib.error
import urllib.request

DEFAULT_CASES = "BUP_CSE_FEST_2026_Participant_Docs/BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"


def post(base_url: str, payload: dict, timeout: float) -> tuple[int, str]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/optimize-energy",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main() -> int:
    cases_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CASES
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    with open(cases_path, encoding="utf-8") as file:
        cases = json.load(file)["cases"]

    latencies, failures = [], 0
    for _ in range(repeats):
        for case in cases:
            started = time.perf_counter()
            status, body = post(base_url, case["input"], timeout=30.0)
            elapsed = time.perf_counter() - started
            ok = status == 200 and len(json.loads(body).get("hourly_plan", [])) == 24
            latencies.append(elapsed)
            if not ok:
                failures += 1
                print(f"FAIL {case['input']['scenario_id']} status={status} {body[:200]}")
            print(f"{case['input']['scenario_id']:<10} {elapsed:6.2f}s status={status}")

    print("\nsamples   ", len(latencies))
    print("failures  ", failures)
    print("median    ", f"{statistics.median(latencies):.2f}s")
    print("p95       ", f"{percentile(latencies, 0.95):.2f}s")
    print("max       ", f"{max(latencies):.2f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
