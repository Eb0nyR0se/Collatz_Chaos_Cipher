# File: heatmap2D.py

import logging
import matplotlib
matplotlib.use('TkAgg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np
import tkinter as tk
from tkinter import messagebox
from matplotlib.widgets import Slider
from cipher_core import signal_spiral_encrypt
from functools import lru_cache
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=level)


@lru_cache(maxsize=None)
def cached_signal_spiral_encrypt(block, key, rounds):
    return signal_spiral_encrypt(block, key, rounds=rounds)


def generate_waveform_heatmap(block, key, rounds=100):
    _, history, waveform = cached_signal_spiral_encrypt(block, key, rounds)

    if not waveform or not isinstance(waveform, list):
        logging.warning("Empty or invalid waveform returned.")
        return np.zeros((10, rounds))

    valid_vals = [int(v) for v in waveform if isinstance(v, (int, float))]
    if not valid_vals:
        logging.warning("Waveform had no numeric values.")
        return np.zeros((10, rounds))

    max_wave = max(valid_vals)
    heat = np.zeros((max_wave + 1, rounds), dtype=int)

    indices = np.array([min(max(int(v), 0), max_wave) for v in waveform])
    rounds_idx = np.arange(len(waveform))
    np.add.at(heat, (indices, rounds_idx), 1)

    return heat


def generate_keysweep_heatmap(block, key_center, key_range, steps=100, rounds=50):
    keys = np.linspace(key_center - key_range, key_center + key_range, steps)
    values_matrix = np.zeros((steps, rounds))

    for i, k in enumerate(keys):
        _, history, _ = cached_signal_spiral_encrypt(block, k, rounds)
        if history and all(isinstance(h, (tuple, list)) and h for h in history):
            values = [float(h[0]) for h in history]
            values_matrix[i, :len(values)] = values

    return keys, values_matrix


def interactive_waveform_heatmap(block, key, max_rounds=100, cmap='inferno', show_colorbar=True):
    heat_full = generate_waveform_heatmap(block, key, rounds=max_rounds)

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)

    cax = ax.imshow(heat_full, aspect='auto', cmap=cmap, origin='lower')
    ax.set_title("Collatz Chaos Cipher Waveform Heatmap")
    ax.set_xlabel("Round Number")
    ax.set_ylabel("Waveform Value (LSB of Block)")

    if show_colorbar:
        fig.colorbar(cax, ax=ax, label="Frequency")

    slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider_rounds = Slider(slider_ax, 'Rounds', 1, max_rounds, valinit=max_rounds, valstep=1)

    def update(val):
        rounds = int(slider_rounds.val)
        new_heat = generate_waveform_heatmap(block, key, rounds=rounds)
        cax.set_data(new_heat)
        cax.set_clim(vmin=new_heat.min(), vmax=new_heat.max())
        ax.set_xlim(0, rounds)
        fig.canvas.draw_idle()

    slider_rounds.on_changed(update)

    return fig, ax


def interactive_keysweep_heatmap(
    block, key_center, initial_key_range=0.1, steps=100,
    max_rounds=50, cmap='viridis', show_colorbar=True, save_path=None
):
    def generate_matrix(rounds, key_range):
        keys = np.linspace(key_center - key_range, key_center + key_range, steps)
        matrix = np.zeros((steps, rounds))
        for i, k in enumerate(keys):
            _, history, _ = cached_signal_spiral_encrypt(block, k, rounds)
            if history and all(isinstance(h, (tuple, list)) and h for h in history):
                values = [float(h[0]) for h in history]
                matrix[i, :len(values)] = values
        return matrix

    values_matrix = generate_matrix(max_rounds, initial_key_range)

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.35)
    cax = ax.imshow(values_matrix, aspect='auto', cmap=cmap, origin='lower')

    ax.set_title("Heatmap of Block Values Over Rounds and Keys")
    ax.set_xlabel("Round")
    ax.set_ylabel("Key Index")

    if show_colorbar:
        fig.colorbar(cax, ax=ax, label='Block Value')

    ax_rounds = plt.axes([0.2, 0.15, 0.6, 0.03])
    ax_keyrange = plt.axes([0.2, 0.05, 0.6, 0.03])

    slider_rounds = Slider(ax_rounds, 'Rounds', 1, max_rounds, valinit=max_rounds, valstep=1)
    slider_keyrange = Slider(ax_keyrange, 'Key Range', 0.001, max(0.5, initial_key_range * 5), valinit=initial_key_range)

    def update(val):
        rounds = int(slider_rounds.val)
        key_range = slider_keyrange.val
        new_matrix = generate_matrix(rounds, key_range)
        cax.set_data(new_matrix)
        cax.set_extent([0, rounds, 0, steps])
        cax.set_clim(vmin=np.min(new_matrix), vmax=np.max(new_matrix))
        ax.set_xlim(0, rounds)
        ax.set_ylim(0, steps)
        fig.canvas.draw_idle()

    slider_rounds.on_changed(update)
    slider_keyrange.on_changed(update)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved keysweep heatmap to {save_path}")

    plt.show()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved keysweep heatmap to {save_path}")

    return fig, ax



def run_heatmap_gui():
    def on_plot():
        try:
            block = float(entry_block.get())
            key = float(entry_key.get())
            mode = var_mode.get()
            cmap = cmap_var.get()

            if frame_plot.winfo_children():
                for child in frame_plot.winfo_children():
                    child.destroy()

            if mode == "waveform":
                fig, ax = interactive_waveform_heatmap(block, key, cmap=cmap)
            else:
                fig, ax = interactive_keysweep_heatmap(block, key, initial_key_range=0.1 * abs(key), cmap=cmap)

            canvas = FigureCanvasTkAgg(fig, master=frame_plot)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)


        except Exception as e:
            messagebox.showerror("Error", str(e))

    root = tk.Tk()
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)

    tk.Label(root, text="Block:").grid(row=0, column=0)
    tk.Label(root, text="Key:").grid(row=1, column=0)
    entry_block = tk.Entry(root)
    entry_key = tk.Entry(root)
    entry_block.insert(0, "12345.6789")
    entry_key.insert(0, "42424242")
    entry_block.grid(row=0, column=1)
    entry_key.grid(row=1, column=1)

    tk.Label(root, text="Mode:").grid(row=2, column=0)
    var_mode = tk.StringVar(value="waveform")
    tk.OptionMenu(root, var_mode, "waveform", "keysweep").grid(row=2, column=1)

    tk.Label(root, text="Colormap:").grid(row=3, column=0)
    cmap_var = tk.StringVar(value="inferno")
    tk.OptionMenu(root, cmap_var, "inferno", "viridis", "plasma", "magma").grid(row=3, column=1)

    tk.Button(root, text="Plot Heatmap", command=on_plot).grid(row=4, columnspan=2, pady=10)

    global frame_plot
    frame_plot = tk.Frame(root)
    frame_plot.grid(row=5, column=0, columnspan=2, sticky="nsew")

    root.mainloop()


if __name__ == "__main__":
    run_heatmap_gui()
