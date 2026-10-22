"""
inject_semantic_profile.py

Injects ADAPT's semantic profile (priority = optimization bias, application_type)
plus the hard/soft requirement fields (data_locality, gpu_acceleration, latency,
compute, availability) into rows of the Alibaba cluster-trace-gpu-v2023 pod list
CSVs, producing entries that can be replayed directly into adapt-scheduler (e.g.
via the Locust SchedulerUser -> MQTT -> main.py path).

Design principle: every injected field is either
  (a) DERIVED from a real trace column (defensible, citable in the TCC), or
  (b) SYNTHETIC but conditioned on a real trace column so it isn't i.i.d. noise, or
  (c) FULLY SYNTHETIC (only data_sensitivity) -- explicitly disclosed as an
      artificial injection with no trace-grounded signal, used purely to
      exercise the data_locality hard-constraint path in benchmarks.

Determinism: every stochastic field is seeded from a hash of (pod name, field
salt), NOT from a global sequential RNG. This means the injection is stable
per-row regardless of processing order, chunking, or re-runs -- important if
you ever process shards in parallel or resume a partial run.

ASSUMPTIONS ABOUT THE JSON-SLA SCHEMA:
The exact field names/enum values below are my best reconstruction of ADAPT's
schema from what you've described (priority, data_sensitivity, application_type,
edge/cloud strategy preference as semantic profile; data_locality, rescheduling,
gpu_acceleration as hard constraints; latency, compute, availability as soft
scoring criteria). Adjust PRIORITY_LEVELS / SENSITIVITY_LEVELS / APP_TYPES /
STRATEGY_PREFS and the JSON shape in `build_sla` to match your actual schema
(and once your co-advisor finalizes it, keep this in sync).
"""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

# ---- schema vocab (edit to match your actual JSON-SLA schema) -------------

# priority = user-declared optimization bias for the application (not an
# urgency/ordinal level). Derived via a precedence cascade -- NOT a 1:1 qos
# map anymore, since high_performance/low_cost are independently determined
# from resource magnitude / estimated instance cost and can apply to pods of
# any qos class. Precedence, as decided with the advisor:
#
#   1. qos == 'Guaranteed'                       -> high_availability
#      (checked FIRST and protected: only 7 pods in the full 8152-pod
#      population carry this qos; every one of them would otherwise be
#      intercepted by rule 2 or 3 below and the class would end up with
#      ZERO representation -- verified empirically before adopting this
#      ordering, see conversation history / methodology notes)
#   2. high_performance rule (GPU/CPU magnitude)  -> high_performance
#   3. low_cost rule (estimated AWS instance cost) -> low_cost
#   4. qos == 'LS'                                -> real_time
#   5. anything left over (BE/Burstable pods that are neither cheap nor
#      GPU/CPU-heavy nor LS -- ~3.5% of the population)
#                                                  -> low_cost (fallback;
#      mirrors BE's original qos-based guess, since these are pods with no
#      stronger claim on performance or latency)
PRIORITY_LEVELS = ["real_time", "low_cost", "high_performance", "high_availability"]

# high_performance CPU threshold for single dedicated-GPU (gpu_milli==1000)
# pods, derived via Otsu's method / Jenks natural breaks (both converged on
# this value against the real openb_pod_list_default.csv distribution) --
# NOT a median split. See methodology notes for the full derivation.
HP_CPU_THRESHOLD_MILLI = 8710

# low_cost threshold in $/hr, derived via Jenks natural breaks on the
# LOG-transformed estimated instance cost (the raw cost distribution is
# heavily right-skewed by multi-GPU instance matches, so an untransformed
# split degenerates into isolating a handful of outliers -- see methodology
# notes). Back-transformed from the log-space break point.
LOW_COST_THRESHOLD_USD_HR = 0.768

# Reference table: (name, vCPU, memory_GiB, GPU_count, on-demand $/hr).
# Pulled Sep 2026, us-east-1, Linux. GPU families kept to one representative
# SKU line per generation (G4dn=T4, G5=A10G, P4d=A100) -- exhaustively
# listing every SKU adds noise, not accuracy, to a cheapest-fit search.
# RE-VERIFY p4d.24xlarge specifically before finalizing: AWS cut P4d/P4de
# on-demand pricing by up to 33% in a 2025 pricing update and this number
# may be stale.
INSTANCE_TABLE = [
    ("m5.large", 2, 8, 0, 0.096), ("m5.xlarge", 4, 16, 0, 0.192),
    ("m5.2xlarge", 8, 32, 0, 0.384), ("m5.4xlarge", 16, 64, 0, 0.768),
    ("m5.8xlarge", 32, 128, 0, 1.536), ("m5.12xlarge", 48, 192, 0, 2.304),
    ("m5.16xlarge", 64, 256, 0, 3.072), ("m5.24xlarge", 96, 384, 0, 4.608),
    ("c5.large", 2, 4, 0, 0.085), ("c5.xlarge", 4, 8, 0, 0.17),
    ("c5.2xlarge", 8, 16, 0, 0.34), ("c5.4xlarge", 16, 32, 0, 0.68),
    ("c5.9xlarge", 36, 72, 0, 1.53), ("c5.12xlarge", 48, 96, 0, 2.04),
    ("c5.18xlarge", 72, 144, 0, 3.06), ("c5.24xlarge", 96, 192, 0, 4.08),
    ("g4dn.xlarge", 4, 16, 1, 0.526), ("g4dn.2xlarge", 8, 32, 1, 0.752),
    ("g4dn.4xlarge", 16, 64, 1, 1.204), ("g4dn.8xlarge", 32, 128, 1, 2.176),
    ("g4dn.12xlarge", 48, 192, 4, 3.912), ("g4dn.16xlarge", 64, 256, 1, 4.352),
    ("g5.xlarge", 4, 16, 1, 1.006), ("g5.2xlarge", 8, 32, 1, 1.212),
    ("g5.4xlarge", 16, 64, 1, 1.624), ("g5.8xlarge", 32, 128, 1, 2.448),
    ("g5.16xlarge", 64, 256, 1, 4.096), ("g5.12xlarge", 48, 192, 4, 5.672),
    ("g5.24xlarge", 96, 384, 4, 8.144), ("g5.48xlarge", 192, 768, 8, 16.288),
    ("p4d.24xlarge", 96, 1152, 8, 32.77),
]


def cheapest_fit_cost(cpu_milli: int, memory_mib: int, num_gpu: int) -> float:
    """Cost of the cheapest AWS instance type that satisfies the request.
    Whole-instance cost, not a fractional GPU-share cost -- AWS doesn't sell
    fractional GPU billing, so a gpu_milli<1000 pod still needs a full GPU
    instance. This means the proxy overestimates cost for GPU-sharing pods
    relative to what they might "really" cost if co-located with others --
    a stated simplification, not an oversight."""
    vcpu_needed = cpu_milli / 1000
    mem_gib_needed = memory_mib / 1024
    candidates = [
        price for _, vcpu, mem, gpu, price in INSTANCE_TABLE
        if vcpu >= vcpu_needed and mem >= mem_gib_needed and gpu >= num_gpu
    ]
    return min(candidates) if candidates else float("inf")


def seeded_rng(pod_name: str, salt: str) -> random.Random:
    """Deterministic per-(pod, field) RNG stream, order-independent."""
    h = hashlib.sha256(f"{pod_name}:{salt}".encode()).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)


def derive_hard_constraints(row: dict) -> dict:
    num_gpu = int(row["num_gpu"])
    qos = row["qos"]

    gpu_acceleration = num_gpu > 0

    # data_locality: synthetic but qos-conditioned -- LS/Guaranteed pods are
    # more likely to require pinned on-prem locality (they represent
    # latency-sensitive services); BE/Burstable are more likely to be
    # locality-agnostic batch. This single field now covers what used to be
    # two separate fields (data_locality / data_sensitivity) -- the platform
    # treats "must stay on-prem" and "is sensitive data" as the same signal.
    rng = seeded_rng(row["name"], "data_locality")
    locality_prob = 0.65 if qos in ("LS", "Guaranteed") else 0.15
    data_locality = rng.random() < locality_prob

    return {
        "gpu_acceleration": gpu_acceleration,
        "data_locality": data_locality,
    }


def derive_application_type(row: dict) -> str:
    num_gpu = int(row["num_gpu"])
    gpu_milli = int(row["gpu_milli"])

    if num_gpu >= 2:
        return "training"
    if num_gpu == 1 and gpu_milli > 0 and gpu_milli < 1000:
        # fractional/shared GPU allocation: characteristic of GPU-sharing
        # inference/serving workloads rather than dedicated training
        return "inference"
    if num_gpu == 1 and gpu_milli >= 1000:
        return "training"
    return "generic"


def is_high_performance(row: dict) -> bool:
    num_gpu = int(row["num_gpu"])
    gpu_milli = int(row["gpu_milli"])
    cpu_milli = int(row["cpu_milli"])
    if num_gpu > 1:
        return True
    if num_gpu == 1 and gpu_milli == 1000 and cpu_milli > HP_CPU_THRESHOLD_MILLI:
        return True
    return False


def derive_priority2(row: dict) -> str:
    qos = row["qos"]
    num_gpu = int(row["num_gpu"])

    # 1. protect the tiny Guaranteed/high_availability class first
    if qos == "Guaranteed":
        return "high_availability"

    # 2. GPU/CPU-magnitude rule
    if is_high_performance(row):
        return "high_performance"

    # 3. estimated-instance-cost rule
    cost = cheapest_fit_cost(int(row["cpu_milli"]), int(row["memory_mib"]), int(row["num_gpu"]))
    if cost <= LOW_COST_THRESHOLD_USD_HR:
        return "low_cost"

    # 4. remaining latency-sensitive pods
    if qos == "LS":
        return "real_time"

    if qos != "BE":
        return "high_availability"  # fallback for any other QoS class that isn't BE/Burstable

    # 5. fallback for whatever's left (BE/Burstable pods that are neither
    # cheap, GPU/CPU-heavy, nor LS -- no stronger claim on performance or
    # latency, so default to low_cost)
    return "low_cost"  # fallback to low_cost for GPU pods that are not high_performance

def derive_priority(row: dict) -> str:
    qos = row["qos"]

    # 1. protect the tiny Guaranteed/high_availability class first
    if qos == "Guaranteed":
        return "high_availability"

    # 2. GPU/CPU-magnitude rule
    if is_high_performance(row):
        return "high_performance"

    # 3. estimated-instance-cost rule
    cost = cheapest_fit_cost(int(row["cpu_milli"]), int(row["memory_mib"]), int(row["num_gpu"]))
    if cost <= LOW_COST_THRESHOLD_USD_HR:
        return "low_cost"

    # 4. remaining latency-sensitive pods
    if qos == "LS":
        return "real_time"

    # 5. fallback for whatever's left (BE/Burstable pods that are neither
    # cheap, GPU/CPU-heavy, nor LS -- no stronger claim on performance or
    # latency, so default to low_cost)
    return "low_cost"


def derive_soft_criteria(row: dict) -> dict:
    qos = row["qos"]
    latency_rng = seeded_rng(row["name"], "latency")
    availability_rng = seeded_rng(row["name"], "availability")

    if qos in ("LS", "Guaranteed"):
        latency_ms = latency_rng.randint(20, 150)
        availability_pct = round(availability_rng.uniform(99.5, 99.99), 2)
    else:
        latency_ms = latency_rng.randint(500, 3000)
        availability_pct = round(availability_rng.uniform(95.0, 99.0), 2)

    # compute: pass through real trace demand instead of synthesizing a score.
    # This is exactly the resources-block extension you flagged as missing
    # in engine.py (currently only carries a hardware_acceleration boolean).
    compute = {
        "cpu_milli": int(row["cpu_milli"]),
        "memory_mib": int(row["memory_mib"]),
        "gpu_milli": int(row["gpu_milli"]),
    }

    return {
        "latency_ms": latency_ms,
        "availability_pct": availability_pct,
        "compute": compute,
    }


def build_sla(row: dict) -> dict:
    hard = derive_hard_constraints(row)
    # soft = derive_soft_criteria(row)
    # app_type = derive_application_type(row)
    priority = derive_priority2(row)

    return {
        "pod_name": row["name"],
        "semantic_profile": {
            "priority": priority,
            # "application_type": app_type,
        },
        "hard_constraints": hard,
        # "soft_criteria": soft,
        "trace_meta": {
            "cpu_milli": int(row["cpu_milli"]),
            "memory_mib": int(row["memory_mib"]),
            "num_gpu": int(row["num_gpu"]),
            "gpu_milli": int(row["gpu_milli"]),
            "source_qos": row["qos"],
            # "pod_phase": row["pod_phase"],
            # "creation_time": int(row["creation_time"]),
            # "deletion_time": int(row["deletion_time"]),
            # "scheduled_time": row["scheduled_time"],
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_csv", type=Path)
    ap.add_argument("--out-csv", type=Path, default=None,
                     help="Write an augmented CSV (original cols + injected cols)")
    ap.add_argument("--out-json-dir", type=Path, default=None,
                     help="Write one JSON-SLA file per pod, ready for MQTT replay")
    args = ap.parse_args()

    with open(args.input_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    slas = [build_sla(row) for row in rows]

    if args.out_csv:
        fieldnames = list(rows[0].keys()) + [
            "priority", "gpu_acceleration",
            "data_locality",
        ]
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row, sla in zip(rows, slas):
                out = dict(row)
                out.update(sla["semantic_profile"])
                out.update(sla["hard_constraints"])
                # out["latency_ms"] = sla["soft_criteria"]["latency_ms"]
                # out["availability_pct"] = sla["soft_criteria"]["availability_pct"]
                writer.writerow(out)
        print(f"Wrote augmented CSV: {args.out_csv} ({len(rows)} rows)")

    if args.out_json_dir:
        args.out_json_dir.mkdir(parents=True, exist_ok=True)
        for sla in slas:
            path = args.out_json_dir / f"{sla['pod_name']}.json"
            path.write_text(json.dumps(sla, indent=2))
        print(f"Wrote {len(slas)} JSON-SLA files to {args.out_json_dir}")

    # quick sanity summary
    from collections import Counter
    print("\n--- label distribution sanity check ---")
    print("priority:", Counter(s["semantic_profile"]["priority"] for s in slas))
    # print("application_type:", Counter(s["semantic_profile"]["application_type"] for s in slas))
    print("data_locality:", Counter(s["hard_constraints"]["data_locality"] for s in slas))


if __name__ == "__main__":
    main()