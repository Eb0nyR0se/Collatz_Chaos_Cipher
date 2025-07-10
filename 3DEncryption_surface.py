# File: 3DEncryption_surface.py

import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from cipher import signal_spiral_encrypt  # Adjust import path as needed

def setup_logging(debug=False):
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        filename='3d_encryption_surface.log',
        filemode='a',
        format='%(asctime)s %(levelname)s: %(message)s',
        level=level,
    )

def validate_positive_int(value, name):
    """Validate positive integer."""
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")

def hamming_distance(a, b):
    """Compute the number of differing bits between two integers."""
    return bin(a ^ b).count('1')

def generate_surface_data(block, key_start, key_end, steps, rounds):
    """
    Generate 3D surface data: block values, waveform LSB, and bit diffusion.

    Returns:
        keys: np.array of keys
        values_matrix: 2D np.array of block values
        waveform_matrix: 2D np.array of waveform LSB values
        bit_diffusion_matrix: 2D np.array of bit-level diffusion values
    """
    keys = np.linspace(key_start, key_end, steps, dtype=np.uint64)
    values_matrix = np.zeros((steps, rounds))
    waveform_matrix = np.zeros((steps, rounds))
    bit_diffusion_matrix = np.zeros((steps, rounds))

    for i, k in enumerate(keys):
        ciphertext, history, waveform = signal_spiral_encrypt(block, int(k), rounds=rounds)
        block_values = [h[0] for h in history]
        values_matrix[i, :] = block_values
        waveform_matrix[i, :] = waveform

        # Compute bit diffusion (Hamming distance between rounds)
        bit_diffusion = [0]  # no previous round for first
        for prev, curr in zip(block_values[:-1], block_values[1:]):
            bit_diffusion.append(hamming_distance(prev, curr))
        bit_diffusion_matrix[i, :] = bit_diffusion

        logging.debug(f"Processed key {k}")

    return keys, values_matrix, waveform_matrix, bit_diffusion_matrix

def plot_surface(keys, values_matrix, waveform_matrix, bit_diffusion_matrix,
                 rounds, color_by='waveform', save_path=None, interactive=False):
    """
    Plot 3D surface of block values colored by waveform or bit diffusion.

    Args:
        keys: np.array of keys
        values_matrix: 2D np.array of block values
        waveform_matrix: 2D np.array of waveform LSB values
        bit_diffusion_matrix: 2D np.array of bit diffusion values
        rounds: number of rounds (int)
        color_by: 'waveform' or 'bit_diffusion'
        save_path: optional file path to save the plot
        interactive: if True, enable interactive plotting
    """
    X, Y = np.meshgrid(np.arange(rounds), np.arange(len(keys)))
    Z = values_matrix

    if color_by == 'bit_diffusion':
        # Normalize bit diffusion for coloring (max possible bits = 64)
        W = bit_diffusion_matrix / 64
        cmap = cm.plasma
        label = 'Bit Diffusion (Normalized Hamming Distance)'
    else:
        # Normalize waveform for coloring
        W = waveform_matrix / 255
        cmap = cm.viridis
        label = 'Waveform LSB Intensity'

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, facecolors=cmap(W), linewidth=0, antialiased=False)

    ax.set_xlabel('Round')
    ax.set_ylabel('Key Index')
    ax.set_zlabel('Block Value')
    ax.set_title(f'3D Surface Plot of Block Values Over Rounds and Keys\n(Color = {label})')

    m = cm.ScalarMappable(cmap=cmap)
    m.set_array(W)
    fig.colorbar(m, shrink=0.5, aspect=10, label=label)

    if interactive:
        plt.ion()
    else:
        plt.ioff()

    if save_path:
        plt.savefig(save_path)
        logging.info(f"3D surface plot saved to {save_path}")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="3D Encryption Surface Visualization for Collatz Chaos Cipher")
    parser.add_argument("--block", type=lambda x: int(x, 0), default=0x123456, help="Plaintext block (default: 0x123456)")
    parser.add_argument("--key-start", type=lambda x: int(x, 0), default=100000, help="Start key range (default: 100000)")
    parser.add_argument("--key-end", type=lambda x: int(x, 0), default=1000000, help="End key range (default: 1000000)")
    parser.add_argument("--steps", type=int, default=200, help="Number of key steps (default: 200)")
    parser.add_argument("--rounds", type=int, default=100, help="Number of rounds (default: 100)")
    parser.add_argument("--color-by", choices=['waveform', 'bit_diffusion'], default='waveform',
                        help="Color surface by waveform LSB intensity or bit diffusion")
    parser.add_argument("--save", type=str, help="Save plot to file (PNG, JPG, etc.)")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    setup_logging(args.debug)

    try:
        validate_positive_int(args.steps, "Steps")
        validate_positive_int(args.rounds, "Rounds")
        if args.key_start >= args.key_end:
            raise ValueError("key-start must be less than key-end")

        keys, values_matrix, waveform_matrix, bit_diffusion_matrix = generate_surface_data(
            args.block, args.key_start, args.key_end, args.steps, args.rounds)

        plot_surface(keys, values_matrix, waveform_matrix, bit_diffusion_matrix,
                     args.rounds, color_by=args.color_by, save_path=args.save, interactive=args.interactive)

    except Exception as e:
        logging.error(f"Error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
