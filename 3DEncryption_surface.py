#File: 3DEncryption_surface.py

import argparse
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from main import signal_spiral_encrypt
from tqdm import tqdm
import csv


# Setup Logging
def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        filename='3d_encryption_surface.log',
        filemode='a',
        format='%(asctime)s %(levelname)s: %(message)s',
        level=level,
    )


# Validators & Helpers
def validate_positive_int(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")

def extract_float(val):
    if isinstance(val, (tuple, list)):
        return extract_float(val[0])
    return float(val)


# Data Generation
def generate_surface_data(block, key_start, key_end, steps, rounds, quiet=False):
    keys = np.linspace(key_start, key_end, steps)
    values_matrix = np.zeros((steps, rounds))
    waveform_matrix = np.zeros((steps, rounds))
    diffusion_matrix = np.zeros((steps, rounds))

    iterator = keys if quiet else tqdm(keys, desc="Generating surface data")
    for i, k in enumerate(iterator):
        ciphertext, history, waveform = signal_spiral_encrypt(block, k, rounds=rounds)
        block_values = [extract_float(h[0]) for h in history]
        values_matrix[i, :] = block_values

        waveform_vals = [extract_float(w) for w in waveform]
        waveform_matrix[i, :] = waveform_vals

        diffusion = [0.0]
        for prev, curr in zip(block_values[:-1], block_values[1:]):
            diff = abs(curr - prev) / (abs(prev) + 1e-9)
            diffusion.append(diff)
        diffusion_matrix[i, :] = diffusion

        logging.debug(f"Processed key {k}")

    return keys, values_matrix, waveform_matrix, diffusion_matrix


# CSV Export
def export_stats_csv(keys, diffusion_matrix, waveform_matrix, filename="stats_export.csv"):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["KeyIndex", "KeyValue", "MeanDiffusion", "MeanWaveform"])
        for i, key in enumerate(keys):
            mean_diff = np.mean(diffusion_matrix[i])
            mean_wave = np.mean(waveform_matrix[i])
            writer.writerow([i, key, mean_diff, mean_wave])
    print(f"Exported stats to {filename}")


# Surface Plot
def plot_surface(keys, values_matrix, waveform_matrix, diffusion_matrix,
                 rounds, color_by='waveform', save_path=None, interactive=False,
                 azim=45, elev=30, colormap_name=None, export_csv=False, animate=False):
    # Apply dark theme before plotting
    mpl.style.use('dark_background')

    x, y = np.meshgrid(np.arange(rounds), np.arange(len(keys)))
    z = values_matrix

    if color_by == 'diffusion':
        w = diffusion_matrix
        default_cmap = 'plasma'
        label = 'Diffusion (Normalized Abs Difference)'
    else:
        w = waveform_matrix / 255.0
        default_cmap = 'viridis'
        label = 'Waveform Intensity'

    cmap_name = colormap_name if colormap_name else default_cmap
    cmap = plt.get_cmap(cmap_name)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    ax.plot_surface(
        x, y, z,
        facecolors=cmap(w),
        linewidth=0.3,
        antialiased=True,
        alpha=0.85
    )

    # Contour projection on XY plane
    ax.contourf(x, y, z, zdir='z', offset=z.min(), cmap=cmap, alpha=0.25)

    # Labels and title
    ax.set_xlabel('Round')
    ax.set_ylabel('Key Index')
    ax.set_zlabel('Block Value')
    ax.set_title(f'3D Surface of Encrypted Block\nColor = {label}  •  View = azim {azim}°, elev {elev}°')
    ax.view_init(elev=elev, azim=azim)

    # Annotate max and min
    max_idx = np.unravel_index(np.argmax(z), z.shape)
    min_idx = np.unravel_index(np.argmin(z), z.shape)
    ax.text(max_idx[1], max_idx[0], z[max_idx], "Max", color='red', fontsize=10, weight='bold')
    ax.text(min_idx[1], min_idx[0], z[min_idx], "Min", color='blue', fontsize=10, weight='bold')

    # Add colorbar
    norm = colors.Normalize(vmin=w.min(), vmax=w.max())
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array(w)
    fig.colorbar(m, ax=ax, shrink=0.5, aspect=12, label=label)

    # Optional export
    if export_csv:
        export_stats_csv(keys, diffusion_matrix, waveform_matrix)

    # Animation
    if animate:
        import matplotlib.animation as animation
        def update(frame):
            ax.view_init(elev=elev, azim=frame)
            return fig,
        ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), blit=False)
        gif_path = save_path if save_path and save_path.lower().endswith('.gif') else '3d_surface_rotation.gif'
        ani.save(gif_path, writer='pillow', fps=20)
        print(f"Saved animation GIF to {gif_path}")
        plt.close(fig)
    elif save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        if interactive:
            plt.ion()
        plt.show()

#Main CLI
def main():
    print("\nVisualizing Encryption Chaos: 3D Surface Tool")
    print("--------------------------------------------------")
    print("You're witnessing a real-time 3D visualization of how a float-based")
    print("encryption algorithm (inspired by the Collatz Conjecture) evolves.")
    print("Each surface point represents a transformed block value across")
    print("multiple rounds and key variations.\n")
    print(" Height = Encrypted block value")
    print(" Color = Either waveform intensity or diffusion")
    print(" Animated view and statistical CSV export are also supported.")
    print("--------------------------------------------------\n")

    parser = argparse.ArgumentParser(description="3D Encryption Surface Visualization for Float Collatz Chaos Cipher")
    parser.add_argument("--block", type=float, default=12345.6789)
    parser.add_argument("--key-start", type=float, default=100000.0)
    parser.add_argument("--key-end", type=float, default=1000000.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--color-by", choices=['waveform', 'diffusion'], default='waveform')
    parser.add_argument("--save", type=str)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--azim", type=float, default=45)
    parser.add_argument("--elev", type=float, default=30)
    parser.add_argument("--colormap", type=str, default=None)
    parser.add_argument("--export-csv", action="store_true")
    parser.add_argument("--animate", action="store_true")
    args = parser.parse_args()

    setup_logging(args.debug)

    try:
        validate_positive_int(args.steps, "Steps")
        validate_positive_int(args.rounds, "Rounds")
        if args.key_start >= args.key_end:
            raise ValueError("key-start must be less than key-end")

        keys, values_matrix, waveform_matrix, diffusion_matrix = generate_surface_data(
            args.block, args.key_start, args.key_end, args.steps, args.rounds, quiet=args.quiet)

        plot_surface(
            keys,
            values_matrix,
            waveform_matrix,
            diffusion_matrix,
            args.rounds,
            color_by=args.color_by,
            save_path=args.save,
            interactive=args.interactive,
            azim=args.azim,
            elev=args.elev,
            colormap_name=args.colormap,
            export_csv=args.export_csv,
            animate=args.animate
        )

    except Exception as e:
        logging.error(f"Error: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
