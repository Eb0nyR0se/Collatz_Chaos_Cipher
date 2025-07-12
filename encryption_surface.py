# File: encryption_surface.py

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mplcursors
from scipy.stats import entropy as scipy_entropy
from cipher_core import signal_spiral_encrypt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk


def export_stats_csv(keys, diffusion_matrix, waveform_matrix, entropy_vector, filename="stats_export.csv"):
    import csv
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["KeyIndex", "KeyValue", "MeanDiffusion", "MeanWaveform", "Entropy"])
        for i, key in enumerate(keys):
            mean_diff = np.mean(diffusion_matrix[i])
            mean_wave = np.mean(waveform_matrix[i])
            writer.writerow([i, key, mean_diff, mean_wave, entropy_vector[i]])
    print(f"Exported stats to {filename}")


def shannon_entropy(signal):
    hist, _ = np.histogram(signal, bins=256, density=True)
    hist = hist + 1e-9  # Avoid log(0)
    return scipy_entropy(hist)


def generate_surface_data(block, key_start, key_end, steps, rounds):
    keys = np.linspace(key_start, key_end, steps)
    values_matrix = np.zeros((steps, rounds))
    waveform_matrix = np.zeros((steps, rounds))
    diffusion_matrix = np.zeros((steps, rounds))
    entropy_vector = []

    for i, k in enumerate(keys):
        ciphertext, history, waveform = signal_spiral_encrypt(block, k, rounds)

        block_values = [float(h[0]) for h in history]
        values_matrix[i, :] = block_values

        waveform_vals = np.array(waveform, dtype=float)
        waveform_matrix[i, :len(waveform_vals)] = waveform_vals

        diffs = np.abs(np.diff(block_values))
        denom = np.abs(block_values[:-1]) + 1e-9
        normalized_diffs = diffs / denom
        diffusion_matrix[i, 1:len(block_values)] = normalized_diffs

        ent = shannon_entropy(waveform_vals)
        entropy_vector.append(ent)

    entropy_vector = np.array(entropy_vector)

    return keys, values_matrix, waveform_matrix, diffusion_matrix, entropy_vector


class Interactive3DSurfaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive 3D Encryption Surface")

        # Default params
        self.block = 12345.6789
        self.key_start = 100000
        self.key_end = 1000000
        self.steps = 50
        self.rounds = 50

        # Plotting option states
        self.color_by = 'waveform'
        self.wavy = False
        self.radial_wave = False
        self.wave_amplitude = 1.0
        self.wave_frequency = 1.0

        # Setup matplotlib figure & 3D axes
        self.fig = plt.figure(figsize=(12, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Add UI widgets inside figure
        self.add_matplotlib_widgets()

        # Embed figure in Tkinter canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Sliders frame
        controls_frame = tk.Frame(self.root)
        controls_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Tkinter sliders
        self.rounds_scale = tk.Scale(controls_frame, from_=10, to=200, orient=tk.HORIZONTAL,
                                     label="Rounds", command=self.slider_changed)
        self.rounds_scale.set(self.rounds)
        self.rounds_scale.pack(fill=tk.X, padx=10, pady=5)

        self.steps_scale = tk.Scale(controls_frame, from_=10, to=200, orient=tk.HORIZONTAL,
                                    label="Steps", command=self.slider_changed)
        self.steps_scale.set(self.steps)
        self.steps_scale.pack(fill=tk.X, padx=10, pady=5)

        # Initial data and plot
        self.update_data()
        self.plot_surface()

    def add_matplotlib_widgets(self):
        # RadioButtons for color_by
        ax_radio = self.fig.add_axes([0.85, 0.6, 0.12, 0.15], facecolor='lightgrey')
        self.radio = RadioButtons(ax_radio, ('waveform', 'diffusion', 'entropy'))
        self.radio.on_clicked(self.on_color_change)
        ax_radio.set_title('Color By')

        # CheckButtons for wavy and radial_wave
        ax_check = self.fig.add_axes([0.85, 0.4, 0.12, 0.15], facecolor='lightgrey')
        self.check = CheckButtons(ax_check, ['Wavy Surface', 'Radial Wave'], [self.wavy, self.radial_wave])
        self.check.on_clicked(self.on_toggle_change)
        ax_check.set_title('Surface Options')

    def on_color_change(self, label):
        self.color_by = label
        self.update_plot()

    def on_toggle_change(self, label):
        if label == 'Wavy Surface':
            self.wavy = not self.wavy
        elif label == 'Radial Wave':
            self.radial_wave = not self.radial_wave
        self.update_plot()

    def slider_changed(self, event):
        self.steps = self.steps_scale.get()
        self.rounds = self.rounds_scale.get()
        self.update_data()
        self.update_plot()

    def update_data(self):
        self.keys, self.values_matrix, self.waveform_matrix, self.diffusion_matrix, self.entropy_vector = generate_surface_data(
            self.block, self.key_start, self.key_end, self.steps, self.rounds
        )

    def plot_surface(self):
        self.ax.clear()

        rounds = self.rounds
        keys = self.keys
        values_matrix = self.values_matrix
        waveform_matrix = self.waveform_matrix
        diffusion_matrix = self.diffusion_matrix
        entropy_vector = self.entropy_vector

        # Meshgrid setup
        if self.radial_wave:
            grid_x, grid_y = rounds, len(keys)
            x_lin = np.linspace(-1, 1, grid_x)
            y_lin = np.linspace(-1, 1, grid_y)
            x, y = np.meshgrid(x_lin, y_lin)
            r = np.sqrt(x ** 2 + y ** 2) + 1e-9
            z = self.wave_amplitude * np.sin(30 * r) * np.exp(-5 * r)
        else:
            x, y = np.meshgrid(np.arange(rounds), np.arange(len(keys)))
            z = values_matrix.copy()
            if self.wavy:
                z_min, z_max = z.min(), z.max()
                wave_scale = self.wave_amplitude * (z_max - z_min) * 0.1
                base_wave = np.sin(2 * np.pi * self.wave_frequency * (x / rounds + y / len(keys)))
                z = z + wave_scale * base_wave

        # Choose coloring data
        if self.color_by == 'diffusion':
            w = diffusion_matrix
            cmap = colors.LinearSegmentedColormap.from_list(
                "pastel_diffusion", ["#87edf7", "#a7abde", "#ffa5d6"], N=256)
            label = 'Diffusion (Normalized Abs Difference)'
        elif self.color_by == 'entropy':
            if self.radial_wave:
                w = z
            else:
                w = np.tile(entropy_vector, (rounds, 1)).T
            cmap = colors.LinearSegmentedColormap.from_list(
                "pastel_entropy", ["#e0ecf4", "#9ebcda", "#8856a7"], N=256)
            label = 'Entropy of Waveform'
        else:
            if self.radial_wave:
                w = z
            else:
                w = waveform_matrix / 255.0
            cmap = colors.LinearSegmentedColormap.from_list(
                "pastel_waveform", ["#ffffcc", "#c2a5cf", "#762a83"], N=256)
            label = 'Waveform Intensity'

        norm = colors.Normalize(vmin=w.min(), vmax=w.max())
        facecolors = cmap(norm(w))

        surf = self.ax.plot_surface(
            x, y, z,
            facecolors=facecolors,
            linewidth=0,
            antialiased=True,
            alpha=0.85,
            rstride=1,
            cstride=1,
        )
        self.ax.plot_wireframe(x, y, z, color='black', linewidth=0.5, rstride=5, cstride=5, alpha=0.7)
        self.ax.contourf(x, y, z, zdir='z', offset=z.min(), cmap=cmap, alpha=0.25)

        self.ax.set_xlabel('Round')
        self.ax.set_ylabel('Key Index')
        self.ax.set_zlabel('Block Value')

        title = f'3D Encryption Surface\nColor = {label}'
        if self.radial_wave:
            title = "3D Radial Wavy Surface (Ripple-like)"
        elif self.wavy:
            title = "3D Wavy Encryption Surface"
        self.ax.set_title(title)

        self.ax.view_init(elev=30, azim=45)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(w)
        self.fig.colorbar(sm, ax=self.ax, shrink=0.5, aspect=12, label=label)

        mplcursors.cursor(surf, hover=True)

        self.canvas.draw_idle()

    def update_plot(self):
        self.plot_surface()


def validate_positive_int(value, name):
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def run_gui():
    root = tk.Tk()
    app = Interactive3DSurfaceApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
