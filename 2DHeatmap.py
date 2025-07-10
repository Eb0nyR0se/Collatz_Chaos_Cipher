# File: 2DHeatmap.py

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from cipher import signal_spiral_encrypt  # ensure these exist

def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=level)

def generate_waveform_heatmap(block, key, rounds=100, modulus=(2**64 - 59)):
    """Generate heatmap data from waveform LSB per round for a single block/key."""
    _, history, waveform = signal_spiral_encrypt(block, key, rounds=rounds, modulus=modulus)
    max_wave = int(max(waveform)) if waveform else 255
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

def interactive_waveform_heatmap(block, key, max_rounds=100, modulus=(2**64 - 59)):
    rounds = max_rounds
    heat = generate_waveform_heatmap(block, key, rounds=rounds, modulus=modulus)

    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.imshow(heat, aspect='auto', cmap='inferno', origin='lower')
    ax.set_title("Collatz Chaos Cipher Waveform Heatmap")
    ax.set_xlabel("Round Number")
    ax.set_ylabel("Waveform Value (LSB of Block)")
    fig.colorbar(cax, ax=ax, label="Frequency")

    ax_round = plt.axes((0.2, 0.05, 0.6, 0.03))  # <- now a tuple!
    slider_rounds = Slider(
        ax=ax_round,
        label='Rounds',
        valmin=1,
        valmax=int(max_rounds),
        valinit=int(rounds),
        valstep=1
    )

    def update(_):
        r = int(slider_rounds.val)
        updated_heat = generate_waveform_heatmap(block, key, rounds=r, modulus=modulus)
        cax.set_data(updated_heat)
        cax.set_clim(vmin=updated_heat.min(), vmax=updated_heat.max())
        ax.set_xlim(0, r)
        fig.canvas.draw_idle()

    slider_rounds.on_changed(update)

    def update(val):
        r = int(val)  # use val directly
        updated_heat = generate_waveform_heatmap(block, key, rounds=r, modulus=modulus)
        cax.set_data(updated_heat)
        cax.set_clim(vmin=updated_heat.min(), vmax=updated_heat.max())
        ax.set_xlim(0, r)
        fig.canvas.draw_idle()
        slider_rounds.on_changed(update)



def heatmap_multiple_keys(block, key_start, key_end,
                          steps=100, rounds=50, modulus=(2 ** 64 - 59)):
    def extract_int(val):
        if isinstance(val, (tuple, list)):
            return extract_int(val[0])
        return int(val)

    ks = extract_int(key_start)
    ke = extract_int(key_end)
    keys = np.linspace(ks, ke, steps, dtype=np.uint64)
    values_matrix = np.zeros((steps, rounds))  # <-- define this before using

    # Debug print once before the loop
    print(f"Type of keys: {type(keys)}")
    print(f"First 5 elements of keys:")
    print(f"ks = {ks} (type {type(ks)})")
    print(f"ke = {ke} (type {type(ke)})")
    print(extract_int((42, 'foo')))  # prints 42
    print(extract_int(((100,), 200)))  # prints 100
    print(extract_int(50))  # prints 50

    for x in keys[:5]:
        print(f"  {x} (type {type(x)})")

    for i, k in enumerate(keys):
        key_int = extract_int(k)  # unwrap tuple if needed
        _, history, _ = signal_spiral_encrypt(block, key_int, rounds=rounds, modulus=modulus)

        values = [extract_int(h[0]) for h in history]
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
    """Convert integer block to list of bits (MSB first)."""
    return [(block >> i) & 1 for i in reversed(range(bit_width))]

def visualize_bit_diffusion(block, key, rounds=16, modulus=(2**64 - 59), save=False):
    """Visualize bit-level diffusion over rounds as a heatmap."""
    _, history, _ = signal_spiral_encrypt(block, key, rounds=rounds, modulus=modulus)

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
    parser.add_argument("--block", type=lambda x: int(x, 0), required=True, help="Plaintext block (hex/int)")
    parser.add_argument("--key", type=lambda x: int(x, 0), help="Key (hex/int) for waveform heatmap")
    parser.add_argument("--rounds", type=int, default=100, help="Number of rounds")
    parser.add_argument("--interactive", action="store_true", help="Interactive slider for rounds (waveform heatmap)")
    parser.add_argument("--multi-key", action="store_true", help="Generate heatmap across multiple keys")
    parser.add_argument("--key-start", type=lambda x: int(x, 0), default=100_000, help="Start key for multi-key mode")
    parser.add_argument("--key-end", type=lambda x: int(x, 0), default=1_000_000, help="End key for multi-key mode")
    parser.add_argument("--bit-diffusion", action="store_true", help="Visualize bit-level diffusion heatmap")
    parser.add_argument("--save", action="store_true", help="Save visualizations instead of showing")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    setup_logging(args.debug)

    if args.bit_diffusion:
        visualize_bit_diffusion(args.block, args.key or 0x4242424242424242, rounds=args.rounds, save=args.save)
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
