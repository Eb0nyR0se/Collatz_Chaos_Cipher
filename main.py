#File: main.pu

import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import csv
from tqdm import tqdm
import tkinter as tk
from tkinter import messagebox
import mplcursors

# Encryption Functions 

def signal_spiral_encrypt(block: float, key: float, rounds: int = 100):
    scale_factor = 10**9
    block_int = int(block * scale_factor)
    key_int = int(key * scale_factor)
    modulus = 2**256

    current = block_int
    history = []
    waveform = []

    for _ in range(rounds):
        if current % 2 == 0:
            current = (current // 2) % modulus
        else:
            current = (3 * current + key_int) % modulus
        current_float = current / scale_factor
        history.append((current_float,))
        waveform.append(current % 256)

    ciphertext = current / scale_factor
    return ciphertext, history, waveform

def signal_spiral_decrypt(ciphertext: float, key: float, rounds: int = 100):
    # Placeholder (actual decryption is non-invertible)
    return ciphertext, [], []

# Helper Functions 

def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        filename='3d_encryption_surface.log',
        filemode='a',
        format='%(asctime)s %(levelname)s: %(message)s',
        level=level,
    )

def validate_positive_int(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")

def extract_float(val):
    if isinstance(val, (tuple, list)):
        return extract_float(val[0])
    return float(val)

def entropy(signal):
    hist, _ = np.histogram(signal, bins=256, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0

# Data Generation 

def generate_surface_data(block, key_start, key_end, steps, rounds, quiet=False):
    keys = np.linspace(key_start, key_end, steps)
    values_matrix = np.zeros((steps, rounds))
    waveform_matrix = np.zeros((steps, rounds))
    diffusion_matrix = np.zeros((steps, rounds))
    entropy_vector = []

    iterator = keys if quiet else tqdm(keys, desc="Generating surface data")
    for i, k in enumerate(iterator):
        ciphertext, history, waveform = signal_spiral_encrypt(block, k, rounds=rounds)
        block_values = [extract_float(h[0]) for h in history]
        values_matrix[i, :] = block_values
        waveform_vals = [extract_float(w) for w in waveform]
        waveform_matrix[i, :] = waveform_vals
        diffusion = [0.0] + [abs(curr - prev) / (abs(prev) + 1e-9) for prev, curr in zip(block_values[:-1], block_values[1:])]
        diffusion_matrix[i, :] = diffusion
        entropy_vector.append(entropy(waveform_vals))

    return keys, values_matrix, waveform_matrix, diffusion_matrix, entropy_vector

# CSV Export 

def export_stats_csv(keys, diffusion_matrix, waveform_matrix, entropy_vector, filename="stats_export.csv"):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["KeyIndex", "KeyValue", "MeanDiffusion", "MeanWaveform", "Entropy"])
        for i, key in enumerate(keys):
            mean_diff = np.mean(diffusion_matrix[i])
            mean_wave = np.mean(waveform_matrix[i])
            writer.writerow([i, key, mean_diff, mean_wave, entropy_vector[i]])
    print(f"Exported stats to {filename}")

# Plotting 

def plot_surface(
    keys, values_matrix, waveform_matrix, diffusion_matrix, entropy_vector, rounds,
    color_by='waveform', save_path=None, interactive=False, azim=45, elev=30,
    colormap_name=None, export_csv=False, animate=False, wavy=False,
    wave_amplitude=1.0, wave_frequency=1.0
):
    plt.style.use('dark_background')
    x, y = np.meshgrid(np.arange(rounds), np.arange(len(keys)))
    z = values_matrix.copy()

    if wavy:
        z_min, z_max = z.min(), z.max()
        wave_scale = wave_amplitude * (z_max - z_min) * 0.1
        wave = wave_scale * np.sin(2 * np.pi * wave_frequency * (x / rounds + y / len(keys)))
        z = z + wave

    if color_by == 'diffusion':
        w = diffusion_matrix
        cmap = LinearSegmentedColormap.from_list("pastel_diffusion", ["#87edf7", "#a7abde", "#ffa5d6"], N=256)
        label = 'Diffusion (Normalized Abs Difference)'
    elif color_by == 'entropy':
        w = np.tile(entropy_vector, (rounds, 1)).T
        cmap = LinearSegmentedColormap.from_list("pastel_entropy", ["#e0ecf4", "#9ebcda", "#8856a7"], N=256)
        label = 'Entropy of Waveform'
    else:
        w = waveform_matrix / 255.0
        cmap = LinearSegmentedColormap.from_list("pastel_waveform", ["#ffffcc", "#c2a5cf", "#762a83"], N=256)
        label = 'Waveform Intensity'

    if colormap_name:
        cmap = plt.get_cmap(colormap_name)
    norm = colors.Normalize(vmin=w.min(), vmax=w.max())
    facecolors = cmap(norm(w))

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        x, y, z,
        facecolors=facecolors,
        linewidth=0,
        antialiased=True,
        alpha=0.85,
        rstride=1,
        cstride=1,
    )
    ax.plot_wireframe(x, y, z, color='black', linewidth=0.5, rstride=5, cstride=5, alpha=0.7)
    ax.contourf(x, y, z, zdir='z', offset=z.min(), cmap=cmap, alpha=0.25)

    max_idx = np.unravel_index(np.argmax(z), z.shape)
    min_idx = np.unravel_index(np.argmin(z), z.shape)
    ax.text(max_idx[1], max_idx[0], z[max_idx], "Max", color='red', fontsize=10, weight='bold')
    ax.text(min_idx[1], min_idx[0], z[min_idx], "Min", color='blue', fontsize=10, weight='bold')

    ax.set_xlabel('Round')
    ax.set_ylabel('Key Index')
    ax.set_zlabel('Block Value')
    ax.set_title(f'3D Encryption Surface\nColor = {label} • View = azim {azim}°, elev {elev}°')
    ax.view_init(elev=elev, azim=azim)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(w)
    fig.colorbar(sm, ax=ax, shrink=0.5, aspect=12, label=label)

    mplcursors.cursor(surf, hover=True)

    if export_csv:
        export_stats_csv(keys, diffusion_matrix, waveform_matrix, entropy_vector)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")

    if interactive:
        plt.ion()
        plt.show(block=True)

    if animate:
        def update(frame):
            ax.view_init(elev=elev, azim=frame)
            return fig,
        ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), blit=False)
        gif_path = save_path if save_path and save_path.lower().endswith('.gif') else '3d_surface_rotation.gif'
        ani.save(gif_path, writer='pillow', fps=20)
        print(f"Saved animation GIF to {gif_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax

# GUI 

def run_gui():
    def on_plot():
        try:
            block = float(entry_block.get())
            key_start = float(entry_start.get())
            key_end = float(entry_end.get())
            steps = int(entry_steps.get())
            rounds = int(entry_rounds.get())
            color_by = var_color.get()
            wavy = var_wavy.get()
            wave_amplitude = float(entry_wave_amp.get())
            wave_frequency = float(entry_wave_freq.get())

            validate_positive_int(steps, "Steps")
            validate_positive_int(rounds, "Rounds")

            keys, val, wave, diff, ent = generate_surface_data(block, key_start, key_end, steps, rounds, quiet=True)
            plot_surface(
                keys, val, wave, diff, ent, rounds,
                color_by=color_by,
                wavy=wavy,
                wave_amplitude=wave_amplitude,
                wave_frequency=wave_frequency
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    root = tk.Tk()
    root.title("3D Encryption Surface Viewer")

    tk.Label(root, text="Block Value:").grid(row=0, column=0, sticky="e")
    tk.Label(root, text="Key Start:").grid(row=1, column=0, sticky="e")
    tk.Label(root, text="Key End:").grid(row=2, column=0, sticky="e")
    tk.Label(root, text="Steps:").grid(row=3, column=0, sticky="e")
    tk.Label(root, text="Rounds:").grid(row=4, column=0, sticky="e")
    tk.Label(root, text="Color By:").grid(row=5, column=0, sticky="e")
    tk.Label(root, text="Wave Amplitude:").grid(row=6, column=0, sticky="e")
    tk.Label(root, text="Wave Frequency:").grid(row=7, column=0, sticky="e")

    entry_block = tk.Entry(root)
    entry_start = tk.Entry(root)
    entry_end = tk.Entry(root)
    entry_steps = tk.Entry(root)
    entry_rounds = tk.Entry(root)
    entry_wave_amp = tk.Entry(root)
    entry_wave_freq = tk.Entry(root)

    entry_block.insert(0, "12345.6789")
    entry_start.insert(0, "100000")
    entry_end.insert(0, "1000000")
    entry_steps.insert(0, "100")
    entry_rounds.insert(0, "50")
    entry_wave_amp.insert(0, "1.0")
    entry_wave_freq.insert(0, "1.0")

    entry_block.grid(row=0, column=1)
    entry_start.grid(row=1, column=1)
    entry_end.grid(row=2, column=1)
    entry_steps.grid(row=3, column=1)
    entry_rounds.grid(row=4, column=1)
    entry_wave_amp.grid(row=6, column=1)
    entry_wave_freq.grid(row=7, column=1)

    var_color = tk.StringVar(value="waveform")
    tk.OptionMenu(root, var_color, "waveform", "diffusion", "entropy").grid(row=5, column=1)

    var_wavy = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Wavy Surface", variable=var_wavy).grid(row=8, columnspan=2, sticky="w")

    tk.Button(root, text="Generate & Plot", command=on_plot).grid(row=9, columnspan=2, pady=10)

    root.mainloop()

# CLI 

def main_cli():
    parser = argparse.ArgumentParser(description="3D Encryption Surface Visualizer")
    subparsers = parser.add_subparsers(dest="command")

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("--block", type=float, default=12345.6789)
    plot_parser.add_argument("--key-start", type=float, default=100000.0)
    plot_parser.add_argument("--key-end", type=float, default=1000000.0)
    plot_parser.add_argument("--steps", type=int, default=200)
    plot_parser.add_argument("--rounds", type=int, default=100)
    plot_parser.add_argument("--color-by", choices=["waveform", "diffusion", "entropy"], default="waveform")
    plot_parser.add_argument("--save", type=str, help="Filename to save plot image")
    plot_parser.add_argument("--interactive", action="store_true", help="Enable interactive matplotlib mode")
    plot_parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    plot_parser.add_argument("--quiet", action="store_true", help="Disable progress bar")
    plot_parser.add_argument("--azim", type=float, default=45, help="Azimuth angle")
    plot_parser.add_argument("--elev", type=float, default=30, help="Elevation angle")
    plot_parser.add_argument("--colormap", type=str, default=None, help="Colormap name")
    plot_parser.add_argument("--export-csv", action="store_true", help="Export statistics CSV")
    plot_parser.add_argument("--animate", action="store_true", help="Generate rotation animation GIF")
    plot_parser.add_argument("--wavy", action="store_true", help="Enable wavy surface modulation")
    plot_parser.add_argument("--wave-amplitude", type=float, default=1.0, help="Wave amplitude")
    plot_parser.add_argument("--wave-frequency", type=float, default=1.0, help="Wave frequency")

    subparsers.add_parser("gui")

    args = parser.parse_args()

    if args.command == "gui" or args.command is None:
        run_gui()
    elif args.command == "plot":
        setup_logging(args.debug)
        try:
            validate_positive_int(args.steps, "Steps")
            validate_positive_int(args.rounds, "Rounds")
            keys, val, wave, diff, ent = generate_surface_data(
                args.block, args.key_start, args.key_end, args.steps, args.rounds, quiet=args.quiet)
            plot_surface(
                keys, val, wave, diff, ent, args.rounds,
                color_by=args.color_by,
                save_path=args.save,
                interactive=args.interactive,
                azim=args.azim,
                elev=args.elev,
                colormap_name=args.colormap,
                export_csv=args.export_csv,
                animate=args.animate,
                wavy=args.wavy,
                wave_amplitude=args.wave_amplitude,
                wave_frequency=args.wave_frequency
            )
        except Exception as e:
            logging.error(f"Error: {e}")
            print(f"Error: {e}")

if __name__ == "__main__":
    main_cli()
