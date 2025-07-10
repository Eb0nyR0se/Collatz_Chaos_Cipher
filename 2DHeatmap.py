import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from cipher_core import signal_spiral_encrypt

def generate_waveform_heatmap(block, key, rounds=100, modulus=(2**64 - 59)):
    """Generate heatmap data from waveform LSB per round for a single block/key."""
    _, history, waveform = signal_spiral_encrypt(block, key, rounds=rounds, modulus=modulus)
    max_wave = max(waveform) if waveform else 255
    heatmap = np.zeros((max_wave + 1, rounds))

    for round_idx, val in enumerate(waveform):
        heatmap[val, round_idx] += 1

    return heatmap

def plot_heatmap(heatmap, title, xlabel, ylabel, cmap='inferno'):
    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.imshow(heatmap, aspect='auto', cmap=cmap, origin='lower')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(cax, ax=ax, label="Frequency")
    plt.tight_layout()
    plt.show()

def interactive_waveform_heatmap(block, key, max_rounds=100, modulus=(2**64 - 59)):
    rounds = max_rounds
    heatmap = generate_waveform_heatmap(block, key, rounds=rounds, modulus=modulus)

    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.imshow(heatmap, aspect='auto', cmap='inferno', origin='lower')
    ax.set_title("Collatz Chaos Cipher Waveform Heatmap")
    ax.set_xlabel("Round Number")
    ax.set_ylabel("Waveform Value (LSB of Block)")
    fig.colorbar(cax, ax=ax, label="Frequency")

    ax_round = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider_rounds = Slider(ax_round, 'Rounds', 1, max_rounds, valinit=rounds, valstep=1)

    def update(val):
        r = int(slider_rounds.val)
        heatmap = generate_waveform_heatmap(block, key, rounds=r, modulus=modulus)
        cax.set_data(heatmap)
        cax.set_clim(vmin=heatmap.min(), vmax=heatmap.max())
        ax.set_xlim(0, r)
        fig.canvas.draw_idle()

    slider_rounds.on_changed(update)
    plt.tight_layout()
    plt.show()

def heatmap_multiple_keys(block, key_start, key_end, steps=100, rounds=50, modulus=(2**64 - 59)):
    keys = np.linspace(key_start, key_end, steps, dtype=np.uint64)
    values_matrix = np.zeros((steps, rounds))

    for i, k in enumerate(keys):
        _, history, _ = signal_spiral_encrypt(block, int(k), rounds=rounds, modulus=modulus)
        values_matrix[i, :] = [h[0] for h in history]

    plt.figure(figsize=(12, 6))
    plt.imshow(values_matrix, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Block Value')
    plt.title("Heatmap of Block Values Over Rounds and Keys")
    plt.xlabel("Round")
    plt.ylabel("Key Index")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Collatz Chaos Cipher Heatmap Visualizations")
    parser.add_argument("--block", type=lambda x: int(x, 0), required=True, help="Plaintext block (hex)")
    parser.add_argument("--key", type=lambda x: int(x, 0), help="Key (hex) for waveform heatmap")
    parser.add_argument("--rounds", type=int, default=100, help="Number of rounds")
    parser.add_argument("--interactive", action="store_true", help="Interactive slider for rounds (waveform heatmap)")
    parser.add_argument("--multi-key", action="store_true", help="Generate heatmap across multiple keys")
    parser.add_argument("--key-start", type=lambda x: int(x, 0), default=100_000, help="Start key (hex/int) for multi-key heatmap")
    parser.add_argument("--key-end", type=lambda x: int(x, 0), default=1_000_000, help="End key (hex/int) for multi-key heatmap")

    args = parser.parse_args()

    if args.multi_key:
        heatmap_multiple_keys(args.block, args.key_start, args.key_end, steps=100, rounds=args.rounds)
    else:
        if args.key is None:
            print("Error: --key required for waveform heatmap visualization")
            return
        if args.interactive:
            interactive_waveform_heatmap(args.block, args.key, max_rounds=args.rounds)
        else:
            heatmap = generate_waveform_heatmap(args.block, args.key, rounds=args.rounds)
            plot_heatmap(heatmap,
                         title="Collatz Chaos Cipher Waveform Heatmap",
                         xlabel="Round Number",
                         ylabel="Waveform Value (LSB of Block)")

if __name__ == "__main__":
    main()
