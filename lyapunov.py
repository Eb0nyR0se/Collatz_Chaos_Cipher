# File: lyapunov.py

import argparse
import logging
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import csv
import numpy as np
from ecdsa import SECP256k1, SigningKey

# Encryption Core with 256-bit modulus, EC key, non-integer Collatz

def derive_key_from_ec(seed: str = "") -> float:
    """
    Generate a pseudo-random key from an EC signing key (SECP256k1),
    converted to float by scaling down large int.
    """
    sk = SigningKey.generate(curve=SECP256k1)
    vk = sk.verifying_key
    key_bytes = vk.to_string()[:16]  # Take first 128 bits (16 bytes)
    key_int = int.from_bytes(key_bytes, 'big')
    # Scale down to float, preserving precision but allowing float math
    return key_int / 1e18


def signal_spiral_encrypt(block: float, key: float, rounds: int = 100):
    """
    Non-integer Collatz-inspired encryption:
    - 256-bit modulus applied to scaled integer values
    - block and key are floats scaled to large integers for mod arithmetic
    - iteration follows Collatz-like rules on large integers
    """

    scale_factor = 1e9  # to preserve decimal precision
    modulus = 2 ** 256

    block_int = int(block * scale_factor)
    key_int = int(key * scale_factor)

    history = []
    waveform = []

    current = block_int

    for _ in range(rounds):
        even = (current % 2 == 0)
        if even:
            current = (current // 2) % modulus
        else:
            current = (3 * current + key_int) % modulus

        current_float = current / scale_factor
        history.append((current_float, even))
        waveform.append(current % 256)  # LSB for waveform

    ciphertext = current / scale_factor
    return ciphertext, history, waveform


def hamming_distance(a, b):
    """Compute the Hamming distance between two integers."""
    return bin(int(a) ^ int(b)).count('1')


def lyapunov_exponent(block, key, rounds=100, delta=1e-9):
    _, history1, _ = signal_spiral_encrypt(block, key, rounds=rounds)
    _, history2, _ = signal_spiral_encrypt(block + delta, key, rounds=rounds)

    distances = []
    for (b1, _), (b2, _) in zip(history1, history2):
        distances.append(abs(b2 - b1))

    distances = np.array(distances)
    distances[distances == 0] = 1e-10  # avoid log(0)

    lyapunov = np.mean(np.log(distances / delta))
    print(f"Approximate Lyapunov exponent: {lyapunov:.5f}")


def visualize_encryption(block, key, rounds=16, save=False, filename="encryption_visual.png",
                         color_even='blue', color_odd='red', color_waveform='purple',
                         verbose=False, export_csv=False, csv_filename="encryption_data.csv"):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format='[%(levelname)s] %(message)s')
    try:
        block = float(block)
        key = float(key)

        logging.info(f"Starting encryption visualization with block={block}, key={key}, rounds={rounds}")

        ciphertext, history, waveform_data = signal_spiral_encrypt(block, key, rounds=rounds)

        values = [float(h[0]) for h in history]
        colors = [color_even if h[1] else color_odd for h in history]
        steps = list(range(1, len(values) + 1))
        bit_diffs = [hamming_distance(values[i], values[i - 1]) if i > 0 else 0 for i in range(len(values))]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)

        ax1.set_title("Collatz Chaos Cipher: Encryption Path")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Block Value")
        ax1.grid(True)
        ax1.plot(steps, values, '-o', color='gray', alpha=0.5)
        for x, y, c in zip(steps, values, colors):
            ax1.scatter(x, y, color=c, s=100)
        ax1.legend(handles=[
            Line2D([0], [0], marker='o', color='w', label='Even', 
                   markerfacecolor=color_even, markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Odd', 
                   markerfacecolor=color_odd, markersize=10)
        ])

        ax2.set_title("Waveform Visualization (LSB of block values)")
        ax2.set_xlabel("Round")
        ax2.set_ylabel("LSB Value")
        ax2.grid(True)
        ax2.plot(steps, waveform_data, marker='o', color=color_waveform)

        ax3.set_title("Bit-Level Diffusion (Hamming Distance Between Rounds)")
        ax3.set_xlabel("Round")
        ax3.set_ylabel("Bit Difference")
        ax3.grid(True)
        ax3.plot(steps, bit_diffs, marker='o', color='orange')

        if save:
            plt.savefig(filename)
            logging.info(f"Visualization saved to {filename}")
        else:
            plt.show()

        if export_csv:
            with open(csv_filename, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Round", "Block Value", "Waveform LSB", "Hamming Distance"])
                for i in range(len(values)):
                    writer.writerow([steps[i], values[i], waveform_data[i], bit_diffs[i]])
            logging.info(f"CSV data exported to {csv_filename}")

        logging.info(f"Encryption visualization completed successfully. Ciphertext: {ciphertext}")

    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        print(f"Error: {e}")


def parse_block_or_key(value: str) -> float:
    value = value.strip().lower()
    return float.fromhex(value) if value.startswith("0x") else float(value)


def main():
    parser = argparse.ArgumentParser(description="Visualize Collatz Chaos Cipher encryption and compute Lyapunov exponent")
    parser.add_argument("--block", type=parse_block_or_key, default="12345.6789",
                        help="Plaintext block (float or hex, default: 12345.6789)")
    parser.add_argument("--key", type=parse_block_or_key, default=None,
                        help="Key (float or hex, default: derived from EC if none provided)")
    parser.add_argument("--rounds", type=int, default=16, help="Number of rounds (default: 16)")
    parser.add_argument("--save", action="store_true", help="Save visualization as PNG")
    parser.add_argument("--filename", type=str, default="encryption_visual.png", help="Output image filename")
    parser.add_argument("--color-even", type=str, default="blue", help="Color for even rounds")
    parser.add_argument("--color-odd", type=str, default="red", help="Color for odd rounds")
    parser.add_argument("--color-waveform", type=str, default="purple", help="Color for waveform plot")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--export-csv", action="store_true", help="Export values to CSV")
    parser.add_argument("--csv-filename", type=str, default="encryption_data.csv", help="CSV output filename")
    parser.add_argument("--lyapunov", action="store_true", help="Compute and print approximate Lyapunov exponent")

    args = parser.parse_args()

    # Derive key from elliptic curve if not given
    key = args.key if args.key is not None else derive_key_from_ec()

    if args.lyapunov:
        lyapunov_exponent(args.block, key, rounds=args.rounds)
    else:
        visualize_encryption(
            block=args.block,
            key=key,
            rounds=args.rounds,
            save=args.save,
            filename=args.filename,
            color_even=args.color_even,
            color_odd=args.color_odd,
            color_waveform=args.color_waveform,
            verbose=args.verbose,
            export_csv=args.export_csv,
            csv_filename=args.csv_filename
        )


if __name__ == "__main__":
    main()
