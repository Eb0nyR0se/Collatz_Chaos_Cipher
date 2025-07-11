# 2DHeatmap.py

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from main import signal_spiral_encrypt
from matplotlib.widgets import Slider
from functools import lru_cache
from tqdm import tqdm


def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=level)


@lru_cache(maxsize=None)
def cached_signal_spiral_encrypt(block, key, rounds):
    return signal_spiral_encrypt(block, key, rounds=rounds)


def generate_waveform_heatmap(block, key, rounds=100):
    _, history, waveform = cached_signal_spiral_encrypt(block, key, rounds)
    max_wave = max(waveform) if waveform else 255
    heat = np.zeros((max_wave + 1, rounds), dtype=int)

    indices = np.array([int(v) for v in waveform])
    rounds_idx = np.arange(len(waveform))
    np.add.at(heat, (indices, rounds_idx), 1)

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
    heat_full = generate_waveform_heatmap(block, key, rounds=max_rounds)

    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.imshow(heat_full, aspect='auto', cmap='inferno', origin='lower')
    ax.set_title("Collatz Chaos Cipher Waveform Heatmap")
    ax.set_xlabel("Round Number")
    ax.set_ylabel("Waveform Value (LSB of Block)")
    fig.colorbar(cax, ax=ax, label="Frequency")

    ax_round = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider_rounds = Slider(
        ax=ax_round,
        label='Rounds',
        valmin=1,
        valmax=int(max_rounds),
        valinit=int(max_rounds),
        valstep=1,
    )

    def update(val):
        r = int(val)
        updated_heat = heat_full[:, :r]
        cax.set_data(updated_heat)
        cax.set_clim(vmin=updated_heat.min(), vmax=updated_heat.max())
        ax.set_xlim(0, r)
        fig.canvas.draw_idle()

    slider_rounds.on_changed(update)
    plt.show()


def heatmap_multiple_keys(block, key_start, key_end, steps=100, rounds=50):
    ks = float(key_start)
    ke = float(key_end)
    keys = np.linspace(ks, ke, steps)
    values_matrix = np.zeros((steps, rounds))

    for i, k in enumerate(tqdm(keys, desc="Generating heatmap for keys")):
        _, history, _ = cached_signal_spiral_encrypt(block, k, rounds)
        values = [float(h[0]) for h in history]
        values_matrix[i, :] = values

    plt.figure(figsize=(12, 6))
    plt.imshow(values_matrix, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Block Value')
    plt.title("Heatmap of Block Values Over Rounds and Keys")
    plt.xlabel("Round")
    plt.ylabel("Key Index")
    plt.tight_layout()
    plt.show()


