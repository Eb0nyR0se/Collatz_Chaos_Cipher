# File: 3DEncryption_surface.py

import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from main import signal_spiral_encrypt  # Make sure this imports your float-based cipher

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

def extract_float(val):
    """Recursively extract a float from nested tuples or lists."""
    if isinstance(val, (tuple, list)):
        return extract_float(val[0])
    return float(val)

def generate_surface_data(block, key_start, key_end, steps, rounds):
    """
    Generate 3D surface data: block values, waveform intensity, and diffusion metric.

    Returns:
        keys: np.array of keys (float)
        values_matrix: 2D np.array of block values (float)
        waveform_matrix: 2D np.array of waveform intensity (float)
        diffusion_matrix: 2D np.array of float diffusion values
    """
    keys = np.linspace(key_start, key_end, steps)  # float keys
    values_matrix = np.zeros((steps, rounds))
    waveform_matrix = np.zeros((steps, rounds))
    diffusion_matrix = np.zeros((steps, rounds))

    for i, k in enumerate(keys):
        ciphertext, history, waveform = signal_spiral_encrypt(block, k, rounds=rounds)

        block_values = [extract_float(h[0]) for h in history]
        values_matrix[i, :] = block_values

        waveform_vals = [extract_float(w) for w in waveform]
        waveform_matrix[i, :] = waveform_vals

        # Diffusion metric for floats: normalized absolute difference between rounds
        diffusion = [0.0]  # no previous round for first
        for prev, curr in zip(block_values[:-1], block_values[1:]):
            diff = abs(curr - prev) / (abs(prev) + 1e-9)  # normalize, avoid div0
            diffusion.append(diff)
        diffusion_matrix[i, :] = diffusion

        logging.debug(f"Processed key {k}")

    return keys, values_matrix, waveform_matrix, diffusion_matrix

def plot_surface(keys, values_matrix, waveform_matrix, diffusion_matrix,
                 rounds, color_by='waveform', save_path=None, interactive=False):
    """
    Plot 3D surface of block values colored by waveform or diffusion.

    Args:
        keys: np.array of keys
        values_matrix: 2D np.array of block values
        waveform_matrix: 2D np.array of waveform intensity
        diffusion_matrix: 2D np.array of diffusion metric
        rounds: number of rounds (int)
        color_by: 'waveform' or 'diffusion'
        save_path: optional file path to save the plot
        interactive: if True, enable interactive plotting
    """
    x, y = np.meshgrid(np.arange(rounds), np.arange(len(keys)))
    z = values_matrix

    if color_by == 'diffusion':
        w = diffusion_matrix
        cmap = cm.get_cmap('plasma')
        label = 'Diffusion (Normalized Abs Difference)'
    else:
        w = waveform_matrix / 255.0  # waveform scaled to [0..1]
        cmap = cm.get_cmap('viridis')
        label = 'Waveform Intensity'

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z, facecolors=cmap(w), linewidth=0, antialiased=False)

    ax.set_xlabel('Round')
    ax.set_ylabel('Key Index')
    ax.set_zlabel('Block Value')
    ax.set_title(f'3D Surface Plot of Block Values Over Rounds and Keys\n(Color = {label})')

    m = cm.ScalarMappable(cmap=cmap)
    m.set_array(w)
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
    parser = argparse.ArgumentParser(description="3D Encryption Surface Visualization for Float Collatz Chaos Cipher")
    parser.add_argument("--block", type=float, default=12345.6789,
                        help="Plaintext block (float, default: 12345.6789)")
    parser.add_argument("--key-start", type=float, default=100000.0,
                        help="Start key range (float, default: 100000.0)")
    parser.add_argument("--key-end", type=float, default=1000000.0,
                        help="End key range (float, default: 1000000.0)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Number of key steps (default: 200)")
    parser.add_argument("--rounds", type=int, default=100,
                        help="Number of rounds (default: 100)")
    parser.add_argument("--color-by", choices=['waveform', 'diffusion'], default='waveform',
                        help="Color surface by waveform intensity or diffusion")
    parser.add_argument("--save", type=str,
                        help="Save plot to file (PNG, JPG, etc.)")
    parser.add_argument("--interactive", action="store_true",
                        help="Enable interactive mode")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()
    setup_logging(args.debug)

    try:
        validate_positive_int(args.steps, "Steps")
        validate_positive_int(args.rounds, "Rounds")
        if args.key_start >= args.key_end:
            raise ValueError("key-start must be less than key-end")

        keys, values_matrix, waveform_matrix, diffusion_matrix = generate_surface_data(
            args.block, args.key_start, args.key_end, args.steps, args.rounds)

        plot_surface(
            keys,
            values_matrix,
            waveform_matrix,
            diffusion_matrix,
            args.rounds,
            color_by=args.color_by,
            save_path=args.save,
            interactive=args.interactive
        )

    except Exception as e:
        logging.error(f"Error: {e}")
        print(f"Error: {e}")

    # Interactive prompt helpers
    def prompt_float(prompt_text, default):
        while True:
            try:
                val = input(f"{prompt_text} [default: {default}]: ").strip()
                if val == "":
                    return default
                return float(val)
            except ValueError:
                print("Invalid input. Please enter a valid float.")

    def prompt_int(prompt_text, default):
        while True:
            try:
                val = input(f"{prompt_text} [default: {default}]: ").strip()
                if val == "":
                    return default
                iv = int(val)
                if iv <= 0:
                    print("Please enter a positive integer.")
                    continue
                return iv
            except ValueError:
                print("Invalid input. Please enter a valid integer.")

    # Use CLI arg or prompt user
    block = args.block if args.block is not None else prompt_float("Enter plaintext block", 12345.6789)
    key_start = args.key_start if args.key_start is not None else prompt_float("Enter start key range", 100000.0)
    key_end = args.key_end if args.key_end is not None else prompt_float("Enter end key range", 1000000.0)
    steps = args.steps if args.steps is not None else prompt_int("Enter number of key steps", 200)
    rounds = args.rounds if args.rounds is not None else prompt_int("Enter number of rounds", 100)

    try:
        validate_positive_int(steps, "Steps")
        validate_positive_int(rounds, "Rounds")
        if key_start >= key_end:
            raise ValueError("key-start must be less than key-end")

        keys, values_matrix, waveform_matrix, diffusion_matrix = generate_surface_data(
            block, key_start, key_end, steps, rounds)

        plot_surface(keys, values_matrix, waveform_matrix, diffusion_matrix,
                     rounds, color_by=args.color_by, save_path=args.save, interactive=args.interactive)

    except Exception as e:
        logging.error(f"Error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
