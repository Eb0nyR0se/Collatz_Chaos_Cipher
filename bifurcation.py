# File: bifurcation.py

import argparse
import matplotlib.pyplot as plt
import numpy as np
from main import signal_spiral_encrypt


def bifurcation_diagram(block, key_start, key_end, steps=200, rounds=30):
    keys = np.linspace(key_start, key_end, steps)  # float keys
    last_values = []

    for k in keys:
        _, history, _ = signal_spiral_encrypt(block, k, rounds=rounds)
        last_values.append(history[-1][0])  # last block value

    plt.figure(figsize=(10, 6))
    plt.plot(keys, last_values, ',k', alpha=0.5)
    plt.title("Bifurcation Diagram (Final value vs. Key)")
    plt.xlabel("Key")
    plt.ylabel("Final Value After Encryption")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate a bifurcation diagram for the Collatz Chaos Cipher")
    parser.add_argument("--block", type=float, default=0x112233,
                        help="Plaintext block value as float (default: 0x112233)")
    parser.add_argument("--key-start", type=float, default=1_000_000.0,
                        help="Start of key range (default: 1,000,000.0)")
    parser.add_argument("--key-end", type=float, default=10_000_000.0,
                        help="End of key range (default: 10,000,000.0)")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of steps between keys (default: 1000)")
    parser.add_argument("--rounds", type=int, default=50,
                        help="Number of encryption rounds per key (default: 50)")

    args = parser.parse_args()

    if args.key_start >= args.key_end:
        print("Error: --key-start must be less than --key-end.")
        return

    bifurcation_diagram(args.block, args.key_start, args.key_end, args.steps, args.rounds)


if __name__ == "__main__":
    main()
