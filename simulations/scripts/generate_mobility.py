#!/usr/bin/env python3
"""Generate synthetic temporal-contact traces for the simulator.

The generator works directly at the contact level. This is intentional: the
simulator needs opportunities to exchange compact knowledge, not geographical
coordinates, radio propagation, or link-layer timing.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line options for a synthetic trace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("edge_markov", "community"), default="edge_markov")
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--duration", type=int, required=True, help="Total simulated time in arbitrary units.")
    parser.add_argument("--time-step", type=int, default=20, help="Spacing between contact snapshots.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-probability", type=float, default=0.10, help="Target marginal probability that one pair is in contact.")
    parser.add_argument("--persistence", type=float, default=0.80, help="Probability that an active edge remains active at the next snapshot.")
    parser.add_argument("--num-communities", type=int, default=3)
    parser.add_argument("--intra-contact-probability", type=float, default=0.35)
    parser.add_argument("--inter-contact-probability", type=float, default=0.02)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate probabilities and the requested timeline before generation."""
    if args.num_nodes < 2:
        raise ValueError("--num-nodes must be at least 2.")
    if args.duration <= 0 or args.time_step <= 0:
        raise ValueError("--duration and --time-step must be positive.")
    if args.num_communities < 1 or args.num_communities > args.num_nodes:
        raise ValueError("--num-communities must be between 1 and --num-nodes.")
    for name in ("contact_probability", "persistence", "intra_contact_probability", "inter_contact_probability"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1].")
    if 0.0 < args.contact_probability < 1.0:
        activation = args.contact_probability * (1.0 - args.persistence) / (1.0 - args.contact_probability)
        if activation > 1.0:
            raise ValueError(
                "The requested contact probability and persistence are incompatible; "
                "increase --persistence or reduce --contact-probability."
            )


def edge_markov_events(args: argparse.Namespace, rng: np.random.Generator) -> list[tuple[int, str, str]]:
    """Generate persistent pairwise contacts with the Edge-Markov model."""
    pairs = list(combinations(range(args.num_nodes), 2))
    if args.contact_probability == 0.0:
        return []
    if args.contact_probability == 1.0:
        return [
            (time, f"trace_{left}", f"trace_{right}")
            for time in range(0, args.duration + 1, args.time_step)
            for left, right in pairs
        ]
    active = rng.random(len(pairs)) < args.contact_probability
    deactivate_probability = 1.0 - args.persistence
    if args.contact_probability in (0.0, 1.0):
        activate_probability = args.contact_probability
    else:
        activate_probability = args.contact_probability * deactivate_probability / (1.0 - args.contact_probability)
    events: list[tuple[int, str, str]] = []
    for time in range(0, args.duration + 1, args.time_step):
        if time > 0:
            keep_active = rng.random(len(pairs)) < args.persistence
            become_active = rng.random(len(pairs)) < activate_probability
            active = np.where(active, keep_active, become_active)
        events.extend((time, f"trace_{left}", f"trace_{right}") for (left, right), is_active in zip(pairs, active, strict=True) if is_active)
    return events


def community_events(args: argparse.Namespace, rng: np.random.Generator) -> tuple[list[tuple[int, str, str]], list[int]]:
    """Generate contacts with stronger within-community than cross-community mixing."""
    communities = np.arange(args.num_nodes) % args.num_communities
    rng.shuffle(communities)
    pairs = list(combinations(range(args.num_nodes), 2))
    events: list[tuple[int, str, str]] = []
    for time in range(0, args.duration + 1, args.time_step):
        for left, right in pairs:
            probability = args.intra_contact_probability if communities[left] == communities[right] else args.inter_contact_probability
            if rng.random() < probability:
                events.append((time, f"trace_{left}", f"trace_{right}"))
    return events, communities.tolist()


def write_trace(path: Path, events: list[tuple[int, str, str]]) -> None:
    """Write simulator-compatible CSV rows in chronological order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time", "node1", "node2"))
        writer.writerows(events)


def write_summary(path: Path, args: argparse.Namespace, events: list[tuple[int, str, str]], communities: list[int] | None) -> None:
    """Save generation parameters and basic trace statistics beside the CSV."""
    unique_pairs = {(node1, node2) for _, node1, node2 in events}
    summary = {
        "model": args.model, "seed": args.seed, "num_nodes": args.num_nodes,
        "duration": args.duration, "time_step": args.time_step,
        "num_contact_events": len(events), "num_unique_pairs": len(unique_pairs),
        "mean_events_per_snapshot": len(events) / (args.duration // args.time_step + 1),
        "parameters": {
            "contact_probability": args.contact_probability, "persistence": args.persistence,
            "num_communities": args.num_communities,
            "intra_contact_probability": args.intra_contact_probability,
            "inter_contact_probability": args.inter_contact_probability,
        },
    }
    if communities is not None:
        summary["community_by_trace_node"] = {f"trace_{index}": int(label) for index, label in enumerate(communities)}
    path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    """Generate a trace and a small reproducibility summary."""
    args = parse_args()
    validate_args(args)
    rng = np.random.default_rng(args.seed)
    if args.model == "edge_markov":
        events, communities = edge_markov_events(args, rng), None
    else:
        events, communities = community_events(args, rng)
    write_trace(args.output, events)
    write_summary(args.output, args, events, communities)
    print(f"Wrote {len(events)} contact events to {args.output}")


if __name__ == "__main__":
    main()
