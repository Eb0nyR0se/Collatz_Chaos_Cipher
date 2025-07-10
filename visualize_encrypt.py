import argparse
import matplotlib.pyplot as plt
import logging
from cipher import signal_spiral_encrypt, rol

def visualize_encryption(block, key, rounds=16, save=False, filename="encryption_visual.png",
                         color_even='blue', color_odd='red', color_waveform='purple', verbose=False):
    """
    Visualize the Collatz Chaos Cipher encryption process.

    Args:
        block (int): The plaintext block to encrypt.
        key (int): The encryption key.
        rounds (int): Number of encryption rounds.
        save (bool): Save plot to a file if True; otherwise display.
        filename (str): Filename to save the visualization image.
        color_even (str): Color for even rounds.
        color_odd (str): Color for odd rounds.
        color_waveform (str): Color for waveform plot.
        verbose (bool): If True, enable verbose logging.
    """

    # Setup logging
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format='[%(levelname)s] %(message)s')

    try:
        # Input validation
        if not (isinstance(block, int) and block >= 0):
            raise ValueError("Block must be a non-negative integer.")
        if not (isinstance(key, int) and key >= 0):
            raise ValueError("Key must be a non-negative integer.")
        if not (isinstance(rounds, int) and rounds > 0):
            raise ValueError("Rounds must be a positive integer.")

        logging.info(f"Starting encryption visualization with block=0x{block:X}, key=0x{key:X}, rounds={rounds}")

        ciphertext, history, waveform_data = signal_spiral_encrypt(block, key, rounds=rounds)

        values = [h[0] for h in history]  # Block values per round
        colors = [color_even if h[1] else color_odd for h in history]  # Even/Odd coloring
        steps = list(range(1, len(values) + 1))

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

        # Encryption Path Plot
        ax1.set_title("Collatz Chaos Cipher: Encryption Path")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Block Value")
        ax1.grid(True)
        ax1.plot(steps, values, '-o', color='gray', alpha=0.5)
        for x, y, c in zip(steps, values, colors):
            ax1.scatter(x, y, color=c, s=100)
        blue_patch = plt.Line2D([0], [0], marker='o', color='w', label='Even', markerfacecolor=color_even, markersize=10)
        red_patch = plt.Line2D([0], [0], marker='o', color='w', label='Odd', markerfacecolor=color_odd, markersize=10)
        ax1.legend(handles=[blue_patch, red_patch])

        # Waveform Plot
        ax2.set_title("Waveform Visualization (LSB of block values)")
        ax2.set_xlabel("Round")
        ax2.set_ylabel("LSB Value")
        ax2.grid(True)
        ax2.plot(steps, waveform_data, marker='o', color=color_waveform)

        # Animation support placeholder (not implemented)
        # Future: Add matplotlib.animation for live encryption visualization

        if save:
            plt.savefig(filename)
            logging.info(f"Visualization saved to {filename}")
        else:
            plt.show()

        logging.info(f"Encryption visualization completed successfully. Ciphertext: 0x{ciphertext:X}")

    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Collatz Chaos Cipher encryption")
    parser.add_argument("--block", type=lambda x: int(x, 0), required=True, help="Plaintext block (hex)")
    parser.add_argument("--key", type=lambda x: int(x, 0), required=True, help="Key (hex)")
    parser.add_argument("--rounds", type=int, default=16, help="Number of rounds")
    parser.add_argument("--save", action="store_true", help="Save visualization as PNG instead of showing it")
    parser.add_argument("--filename", type=str, default="encryption_visual.png", help="Filename to save image")
    parser.add_argument("--color-even", type=str, default="blue", help="Color for even rounds")
    parser.add_argument("--color-odd", type=str, default="red", help="Color for odd rounds")
    parser.add_argument("--color-waveform", type=str, default="purple", help="Color for waveform plot")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    visualize_encryption(args.block, args.key, rounds=args.rounds, save=args.save,
                         filename=args.filename, color_even=args.color_even, color_odd=args.color_odd,
                         color_waveform=args.color_waveform, verbose=args.verbose)

if __name__ == "__main__":
    main()
