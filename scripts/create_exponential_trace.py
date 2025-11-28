#!/usr/bin/env python3
"""
Create a new trace with exponentially distributed output lengths.

Usage:
    python scripts/create_exponential_trace.py azure_chat_23
"""

import argparse
import sys
import os
import numpy as np

# Add parent directory to path to import Dataset module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Dataset.dataset import Requests, ArrivalTimes, Request


def create_exponential_trace(original_name: str, seed: int = 42):
    """
    Create a new trace with:
    - Same arrival times
    - Same input lengths
    - Output lengths drawn i.i.d. from exponential distribution with mean = original mean

    Args:
        original_name: Name of the original trace (e.g., 'azure_chat_23')
        seed: Random seed for reproducibility
    """
    # Set random seed
    np.random.seed(seed)

    # Load original trace
    print(f"Loading original trace: {original_name}")
    original_requests = Requests.load(original_name)
    original_arrivals = ArrivalTimes.load(original_name)

    # Compute mean output length
    output_lengths = [req.output_length for req in original_requests.requests]
    mean_output_length = np.mean(output_lengths)
    print(f"Original mean output length: {mean_output_length:.2f}")
    print(f"Number of requests: {len(original_requests.requests)}")

    # Generate new output lengths from exponential distribution
    # For exponential distribution: mean = scale parameter
    new_output_lengths = np.random.exponential(scale=mean_output_length,
                                                size=len(original_requests.requests))
    # Round to integers
    new_output_lengths = np.round(new_output_lengths).astype(int)
    # Ensure at least 1 token output
    new_output_lengths = np.maximum(new_output_lengths, 1)

    print(f"New mean output length: {np.mean(new_output_lengths):.2f}")

    # Create new requests with modified output lengths
    new_requests = []
    for original_req, new_output_len in zip(original_requests.requests, new_output_lengths):
        new_req = Request(
            input_length=original_req.input_length,
            output_length=int(new_output_len),
            cached_length=original_req.cached_length,
            prompt=original_req.prompt,
            thinking=original_req.thinking,
            answer=original_req.answer,
            thinking_length=original_req.thinking_length
        )
        new_requests.append(new_req)

    # Create new trace name
    new_name = f"ExpD_{original_name}"

    # Create new Requests and ArrivalTimes objects
    new_requests_obj = Requests(
        name=new_name,
        requests=new_requests,
        is_reasoning=original_requests.is_reasoning
    )

    new_arrivals_obj = ArrivalTimes(
        name=new_name,
        arrival_times=original_arrivals.arrival_times.copy()
    )

    # Save the new traces
    print(f"\nSaving new trace: {new_name}")
    new_requests_obj.save()
    new_arrivals_obj.save()

    print(f"\nSuccessfully created exponential trace: {new_name}")
    print(f"Files created:")
    print(f"  - {new_name}.requests.pkl")
    print(f"  - {new_name}.arrival.pkl")

    return new_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a new trace with exponentially distributed output lengths"
    )
    parser.add_argument(
        "trace_name",
        type=str,
        help="Name of the original trace (e.g., azure_chat_23)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()
    create_exponential_trace(args.trace_name, args.seed)
