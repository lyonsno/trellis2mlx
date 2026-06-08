"""Benchmark sequential vs parallel pipeline throughput.

Usage:
    PYTHONPATH=. python bench_parallel.py --image assets/shoe_input.png
    PYTHONPATH=. python bench_parallel.py --image assets/shoe_input.png --parallel 3
"""

import argparse
import os
import subprocess
import time


def run_one(image, seed, output, label=""):
    """Run a single pipeline, return wall time."""
    cmd = [
        "python", "generate.py",
        "--image", image,
        "--seed", str(seed),
        "--output", output,
    ]
    env = {**os.environ, "PYTHONPATH": "."}
    t0 = time.perf_counter()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  {label} FAILED: {result.stderr[-200:]}")
    return elapsed, result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--parallel", type=int, default=2,
                        help="Number of parallel processes (default: 2)")
    parser.add_argument("--seeds", default="42,123",
                        help="Comma-separated seeds")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    n = min(args.parallel, len(seeds))

    # === Sequential ===
    print(f"=== Sequential ({n} runs) ===")
    seq_times = []
    for i in range(n):
        output = f"/tmp/trellis-bench-seq-{seeds[i]}.glb"
        t, ok = run_one(args.image, seeds[i], output, f"seed={seeds[i]}")
        status = "ok" if ok else "FAIL"
        print(f"  seed={seeds[i]}: {t:.1f}s ({status})")
        seq_times.append(t)
    seq_total = sum(seq_times)
    print(f"  Sequential total: {seq_total:.1f}s")

    # === Parallel ===
    print(f"\n=== Parallel ({n} processes) ===")
    procs = []
    env = {**os.environ, "PYTHONPATH": "."}
    t0 = time.perf_counter()
    for i in range(n):
        output = f"/tmp/trellis-bench-par-{seeds[i]}.glb"
        cmd = [
            "python", "generate.py",
            "--image", args.image,
            "--seed", str(seeds[i]),
            "--output", output,
        ]
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append((seeds[i], p))

    # Wait for all
    for seed, p in procs:
        p.wait()
        status = "ok" if p.returncode == 0 else "FAIL"
        print(f"  seed={seed}: {status}")

    par_total = time.perf_counter() - t0
    print(f"  Parallel wall time: {par_total:.1f}s")

    # === Summary ===
    print(f"\n=== Summary ===")
    print(f"  Sequential: {seq_total:.1f}s ({n} runs)")
    print(f"  Parallel:   {par_total:.1f}s ({n} processes)")
    print(f"  Speedup:    {seq_total/par_total:.2f}x")
    print(f"  Efficiency: {seq_total/par_total/n*100:.0f}%")


if __name__ == "__main__":
    main()
