# main.py

import argparse
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm
import csv

mpl.style.use('dark_background')

def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        filename='3d_encryption_surface.log',
        filemode='a',
        format='%(asctime)s %(levelname)s: %(message)s',
        level=level,
    )

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
    # Placeholder - decryption not implemented
    return ciphertext, [], []

def validate_positive_int(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")

def extract_float(val):
    if isinstance(val, (tuple, list)):
        return extract_float(val[0])
    return float(val)

def generate_surface_data(block, key_start, key_end, steps, rounds, quiet=False):
    keys = np.linspace(key_start, key_end, steps)
    values_matrix = np.zeros((steps, rounds))
    waveform_matrix = np.zeros((steps, rounds))
    diffusion_matrix = np.zeros((steps, rounds))

    iterator = keys if quiet else tqdm(keys, desc="Generating surface data")
    for i, k in enumerate(iterator):
        k = float(k)
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

def export_stats_csv(keys, diffusion_matrix, waveform_matrix, filename="stats_export.csv"):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["KeyIndex", "KeyValue", "MeanDiffusion", "MeanWaveform"])
        for i, key in enumerate(keys):
            mean_diff = np.mean(diffusion_matrix[i])
            mean_wave = np.mean(waveform_matrix[i])
            writer.writerow([i, key, mean_diff, mean_wave])
    print(f"Exported stats to {filename}")

def plot_surface(keys, values_matrix, waveform_matrix, diffusion_matrix,
                 rounds, color_by='waveform', save_path=None, interactive=False,
                 azim=45, elev=30, colormap_name=None, export_csv=False, animate=False):

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

    surf = ax.plot_surface(
        x, y, z,
        facecolors=cmap(w),
        linewidth=0.3,
        antialiased=True,
        alpha=0.85
    )

    ax.contourf(x, y, z, zdir='z', offset=z.min(), cmap=cmap, alpha=0.25)

    ax.set_xlabel('Round')
    ax.set_ylabel('Key Index')
    ax.set_zlabel('Block Value')
    ax.set_title(f'3D Surface of Encrypted Block\nColor = {label}  •  View = azim {azim}°, elev {elev}°')
    ax.view_init(elev=elev, azim=azim)

    max_idx = np.unravel_index(np.argmax(z), z.shape)
    min_idx = np.unravel_index(np.argmin(z), z.shape)
    ax.text(max_idx[1], max_idx[0], z[max_idx], "Max", color='red', fontsize=10, weight='bold')
    ax.text(min_idx[1], min_idx[0], z[min_idx], "Min", color='blue', fontsize=10, weight='bold')

    norm = colors.Normalize(vmin=w.min(), vmax=w.max())
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array(w)
    fig.colorbar(m, ax=ax, shrink=0.5, aspect=12, label=label)

    if export_csv:
        export_stats_csv(keys, diffusion_matrix, waveform_matrix)

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

def main():
    setup_logging(False)  # or True if you want debug by default
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
    print("Choose an option:")
    print("1) Encrypt")
    print("2) Decrypt")
    print("3) Visualize")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == '1':
        block = float(input("Enter block (float): "))
        key = float(input("Enter key (float): "))
        rounds = int(input("Enter rounds (default 100): ") or "100")
        ciphertext, _, _ = signal_spiral_encrypt(block, key, rounds=rounds)
        print(f"Ciphertext: {ciphertext}")
    elif choice == '2':
        ciphertext = float(input("Enter ciphertext (float): "))
        key = float(input("Enter key (float): "))
        rounds = int(input("Enter rounds (default 100): ") or "100")
        plaintext, _, _ = signal_spiral_decrypt(ciphertext, key, rounds=rounds)
        print(f"Decrypted (approx.): {plaintext}")
    elif choice == '3':
        block = float(input("Enter block (float, default 12345.6789): ") or "12345.6789")
        key_start = float(input("Enter key start (default 100000): ") or "100000")
        key_end = float(input("Enter key end (default 1000000): ") or "1000000")
        steps = int(input("Enter steps (default 200): ") or "200")
        rounds = int(input("Enter rounds (default 100): ") or "100")
        keys, values_matrix, waveform_matrix, diffusion_matrix = generate_surface_data(
            block, key_start, key_end, steps, rounds, quiet=False
        )
        plot_surface(keys, values_matrix, waveform_matrix, diffusion_matrix, rounds=rounds)
    else:
        print("Invalid choice, exiting.")

if __name__ == "__main__":
    main()
