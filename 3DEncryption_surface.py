import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from cipher_core import signal_spiral_encrypt  # Adjust import path as needed

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

def generate_surface_data(block, key_start, key_end, steps, rounds):
    """
    Generate 3D surface data and waveform data for the cipher.

    Returns:
        keys: np.array of keys
        values_matrix: 2D np.array of block values
        waveform_matrix: 2D np.array of waveform data (LSB per round)
    """
    keys = np.linspace(key_start, key_end, steps, dtype=np.uint64)
    values_matrix = np.zeros((steps, rounds))
    waveform_matrix = np.zeros((steps, rounds))

    for i, k in enumerate(keys):
        ciphertext, history, waveform = signal_spiral_encrypt(block, int(k), rounds=rounds)
        values_matrix[i, :] = [h[0] for h in history]
        waveform_matrix[i, :] = waveform
        logging.debug(f"Processed key {k}")

    return keys, values_matrix, waveform_matrix

def plot_surface(keys, values_matrix, waveform_matrix, rounds, save_path=None, interactive=False):
    """
    Plot 3D surface of block values colored by waveform LSB intensity.

    Args:
        keys: np.array of keys
        values_matrix: 2D np.array of block values
        waveform_matrix: 2D np.array of waveform LSB values
        rounds: number of rounds (int)
        save_path: optional file path to save the plot
        interactive: if True, enable interactive plotting
    """
    X, Y = np.meshgrid(np.arange(rounds), np.arange(len(keys)))
    Z = values_matrix
    W = waveform_matrix / 255  # normalize for colormap

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, facecolors=cm.viridis(W), linewidth=0, antialiased=False)

    ax.set_xlabel('Round')
    ax.set_ylabel('Key Index')
    ax.set_zlabel('Block Value')
    ax.set_title('3D Surface Plot of Block Values Over Rounds and Keys\n(Color = Waveform LSB Intensity)')

    m = cm.ScalarMappable(cmap=cm.viridis)
    m.set_array(W)
    fig.colorbar(m, shrink=0.5, aspect=10, label='Waveform LSB Intensity')

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

        keys, values_matrix, waveform_matrix = generate_surface_data(
            args.block, args.key_start, args.key_end, args.steps, args.rounds)

        plot_surface(keys, values_matrix, waveform_matrix, args.rounds,
                     save_path=args.save, interactive=args.interactive)

    except Exception as e:
        logging.error(f"Error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
