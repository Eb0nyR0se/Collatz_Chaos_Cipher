# File: 2DHeatmap.py

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from cipher import signal_spiral_encrypt  # must be float-compatible cipher

def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=level)

def generate_waveform_heatmap(block, key, rounds=100):
    """Generate heatmap data from waveform LSB per round for a single block/key."""
    _, history, waveform = signal_spiral_encrypt(block, key, rounds=rounds)
    if not waveform:
        max_wave = 255
    else:
        max_wave = int(max(waveform))
    heat = np.zeros((max_wave + 1, rounds))

    for round_idx, val in enumerate(waveform):
        heat[int(val), round_idx] += 1

    return heat

def plot_heatmap(heat, title, xlabel, ylabel, cmap='inferno'):
    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.imshow(heat, aspect='auto', cmap=cmap, origin='lower')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(cax, ax=ax, label="Frequency")
    plt.tight_layout()
    plt.show()

def interactive_waveform_heatmap(block, key, max_rounds=100):
    rounds = max_rounds
    heat = generate_waveform_heatmap(block, key, rounds=rounds)

    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.imshow(heat, aspect='auto', cmap='inferno', origin='lower')
    ax.set_title("Collatz Chaos Cipher Waveform Heatmap")
    ax.set_xlabel("Round Number")
    ax.set_ylabel("Waveform Value (LSB of Block)")
    fig.colorbar(cax, ax=ax, label="Frequency")

    ax_round = plt.axes((0.2, 0.05, 0.6, 0.03))
    slider_rounds = Slider(
        ax=ax_round,
        label='Rounds',
        valmin=1,
        valmax=int(max_rounds),
        valinit=int(rounds),
        valstep=1
    )

    def update(val):
        r = int(val)
        updated_heat = generate_waveform_heatmap(block, key, rounds=r)
        cax.set_data(updated_heat)
        cax.set_clim(vmin=updated_heat.min(), vmax=updated_heat.max())
        ax.set_xlim(0, r)
        fig.canvas.draw_idle()

    slider_rounds.on_changed(update)
    plt.show()

def heatmap_multiple_keys(block, key_start, key_end, steps=100, rounds=50):
    def extract_float(val):
        if isinstance(val, (tuple, list)):
            return extract_float(val[0])
        return float(val)

    ks = extract_float(key_start)
    ke = extract_float(key_end)
    keys = np.linspace(ks, ke, steps)
    values_matrix = np.zeros((steps, rounds))

    for i, k in enumerate(keys):
        _, history, _ = signal_spiral_encrypt(block, k, rounds=rounds)

        values = [extract_float(h[0]) for h in history]
        values_matrix[i, :] = values

    plt.figure(figsize=(12, 6))
    plt.imshow(values_matrix, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Block Value')
    plt.title("Heatmap of Block Values Over Rounds and Keys")
    plt.xlabel("Round")
    plt.ylabel("Key Index")
    plt.tight_layout()
    plt.show()

def block_to_bits(block, bit_width=64):
    """Convert integer block to list of bits (MSB first). Works only if block is int."""
    if not isinstance(block, int):
        raise TypeError("block_to_bits only supports integer blocks.")
    return [(block >> i) & 1 for i in reversed(range(bit_width))]

def visualize_bit_diffusion(block, key, rounds=16, save=False):
    """Visualize bit-level diffusion over rounds as a heatmap."""
    # Only meaningful if block and keys are integers
    if not isinstance(block, int) or not isinstance(key, int):
        print("Bit diffusion visualization only supported for integer block/key.")
        return

    _, history, _ = signal_spiral_encrypt(block, key, rounds=rounds)

    bit_matrix = []
    for state, _, _ in history:
        bits = block_to_bits(state)
        bit_matrix.append(bits)

    bit_matrix = np.array(bit_matrix)

    plt.figure(figsize=(12, 6))
    plt.title("Bit-Level Diffusion Over Encryption Rounds")
    plt.xlabel("Bit Position (MSB to LSB)")
    plt.ylabel("Round")
    plt.imshow(bit_matrix, cmap='Greys', interpolation='nearest', aspect='auto', origin='upper')
    plt.colorbar(label='Bit Value')
    plt.tight_layout()

    if save:
        plt.savefig("bit_diffusion.png")
        print("Saved bit diffusion visualization to bit_diffusion.png")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Collatz Chaos Cipher Heatmap Visualizations")
    parser.add_argument("--block", type=float, required=True, help="Plaintext block (float)")
    parser.add_argument("--key", type=float, help="Key (float) for waveform heatmap")
    parser.add_argument("--rounds", type=int, default=100, help="Number of rounds")
    parser.add_argument("--interactive", action="store_true", help="Interactive slider for rounds (waveform heatmap)")
    parser.add_argument("--multi-key", action="store_true", help="Generate heatmap across multiple keys")
    parser.add_argument("--key-start", type=float, default=100_000.0, help="Start key for multi-key mode")
    parser.add_argument("--key-end", type=float, default=1_000_000.0, help="End key for multi-key mode")
    parser.add_argument("--bit-diffusion", action="store_true", help="Visualize bit-level diffusion heatmap (int only)")
    parser.add_argument("--save", action="store_true", help="Save visualizations instead of showing")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    setup_logging(args.debug)

    if args.bit_diffusion:
        visualize_bit_diffusion(args.block, int(args.key) if args.key else 0x4242424242424242, rounds=args.rounds, save=args.save)
        return

    if args.multi_key:
        heatmap_multiple_keys(args.block, args.key_start, args.key_end, steps=100, rounds=args.rounds)
        return

    if args.key is None:
        print("Error: --key is required unless using --multi-key or --bit-diffusion.")
        return

    if args.interactive:
        interactive_waveform_heatmap(args.block, args.key, max_rounds=args.rounds)
    else:
        heat = generate_waveform_heatmap(args.block, args.key, rounds=args.rounds)
        plot_heatmap(heat,
                     title="Collatz Chaos Cipher Waveform Heatmap",
                     xlabel="Round Number",
                     ylabel="Waveform Value (LSB of Block)")

if __name__ == "__main__":
    main()
