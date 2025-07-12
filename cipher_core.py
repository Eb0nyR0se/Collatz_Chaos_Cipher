# File: cipher_core.py

import subprocess
import sys
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg' depending on your OS
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from ecdsa import SECP256k1, SigningKey

# ------------------------ Encryption Core ------------------------

def signal_spiral_encrypt(block: float, key: float, rounds: int = 100):
    history = []
    waveform = []

    scale_factor = 10 ** 9
    block_int = int(block * scale_factor)
    key_int = int(key * scale_factor)
    modulus = 2 ** 256

    current = block_int
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
    return ciphertext  # Placeholder for future decryption logic

# ------------------------ EC Key Derivation ------------------------

def derive_key_from_ec(seed: str = "") -> float:
    sk = SigningKey.generate(curve=SECP256k1)
    vk = sk.verifying_key
    key_bytes = vk.to_string()[:16]
    key_int = int.from_bytes(key_bytes, 'big')
    return key_int / 1e9

# ------------------------ Encryption Handlers ------------------------

def handle_encrypt(block, key, rounds):
    ciphertext, _, _ = signal_spiral_encrypt(block, key, rounds=rounds)
    print(f"\nEncrypted block {block} with key {key} → Ciphertext: {ciphertext}\n")

def handle_decrypt(block, key, rounds):
    plaintext = signal_spiral_decrypt(block, key, rounds=rounds)
    print(f"\nDecryption not reversible, placeholder result: {plaintext}\n")

# ------------------------ Bifurcation Logic ------------------------

def bifurcation_diagram(block, key_start, key_end, steps=200, rounds=30, workers=4, use_ec_seed=False, ec_seed_str=""):
    if use_ec_seed:
        base_key = derive_key_from_ec(ec_seed_str)
        keys = np.linspace(base_key + key_start, base_key + key_end, steps)
    else:
        keys = np.linspace(key_start, key_end, steps)

    last_values = [None] * steps

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(signal_spiral_encrypt, block, k, rounds): i
            for i, k in enumerate(keys)
        }
        for future in tqdm(as_completed(futures), total=steps, desc="Calculating bifurcation values"):
            i = futures[future]
            try:
                _, history, _ = future.result()
                last_values[i] = history[-1][0] if history else float('nan')
            except Exception as e:
                print(f"Error at key index {i}: {e}")
                last_values[i] = float('nan')

    return keys, last_values

def plot_bifurcation(keys, values):
    plt.figure(figsize=(10, 6))
    plt.plot(keys, values, ',k', alpha=0.5)
    plt.title("Bifurcation Diagram (Final value vs. Key)")
    plt.xlabel("Key")
    plt.ylabel("Final Value After Encryption")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------ Interactive Bifurcation Visualization ------------------------

def interactive_bifurcation():
    """
    Interactive bifurcation diagram with sliders to control parameters.
    Shows the bifurcation plot and updates on slider changes.
    Starts zoomed on the middle 50% of the key range.
    """

    # Default parameters
    block = 0.5
    key_start = 0.0
    key_end = 1.0
    steps = 560
    rounds = 48

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, bottom=0.35)

    # Compute initial bifurcation data
    keys, values = bifurcation_diagram(block, key_start, key_end, steps, rounds)

    # Filter out invalid values (NaN, inf)
    filtered_values = np.array(values)
    filtered_values = filtered_values[np.isfinite(filtered_values)]
    if len(filtered_values) == 0:
        filtered_values = np.array([0, 1])  # fallback to avoid empty ylim

    # Plot line
    line, = ax.plot(keys, values, '.', color='black', alpha=0.9, markersize=3)
    ax.set_title("Bifurcation Diagram (Final value vs. Key)")
    ax.set_xlabel("Key")
    ax.set_ylabel("Final Value After Encryption")
    ax.grid(True)

    # Compute limits for middle 50% zoom on initial plot
    mid_start_idx = len(keys) // 4
    mid_end_idx = 3 * len(keys) // 4
    x_min = keys[mid_start_idx]
    x_max = keys[mid_end_idx]

    y_subset = filtered_values[mid_start_idx:mid_end_idx]
    y_min = np.min(y_subset)
    y_max = np.max(y_subset)
    y_pad = (y_max - y_min) * 0.1
    if y_pad == 0:
        y_pad = 0.1  # minimal padding to avoid flat ylim

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Slider axes
    ax_block = plt.axes([0.1, 0.25, 0.8, 0.03])
    ax_key_start = plt.axes([0.1, 0.2, 0.8, 0.03])
    ax_key_end = plt.axes([0.1, 0.15, 0.8, 0.03])
    ax_steps = plt.axes([0.1, 0.1, 0.8, 0.03])
    ax_rounds = plt.axes([0.1, 0.05, 0.8, 0.03])

    # Create sliders
    slider_block = Slider(ax_block, 'Block', 0.0, 1.0, valinit=block, valstep=0.01)
    slider_key_start = Slider(ax_key_start, 'Key Start', 0.0, 10.0, valinit=key_start)
    slider_key_end = Slider(ax_key_end, 'Key End', 0.1, 20.0, valinit=key_end)
    slider_steps = Slider(ax_steps, 'Steps', 50, 1000, valinit=steps, valstep=10)
    slider_rounds = Slider(ax_rounds, 'Rounds', 10, 100, valinit=rounds, valstep=1)

    def update(val):
        # Read slider values
        b = slider_block.val
        ks = slider_key_start.val
        ke = slider_key_end.val
        st = int(slider_steps.val)
        rd = int(slider_rounds.val)

        # Clamp key_end to be strictly greater than key_start
        if ke <= ks:
            ke = ks + 0.1
            slider_key_end.set_val(ke)

        # Calculate bifurcation values
        print(f"Calculating bifurcation for block={b}, key range=({ks}, {ke}), steps={st}, rounds={rd}")
        keys, values = bifurcation_diagram(b, ks, ke, steps=st, rounds=rd)

        # Filter out invalid values
        filtered_values = np.array(values)
        filtered_values = filtered_values[np.isfinite(filtered_values)]
        if len(filtered_values) == 0:
            print("Warning: No valid bifurcation values found. Skipping update.")
            return

        # Update plot data and axis limits
        line.set_data(keys, values)
        ax.set_xlim(ks, ke)
        ax.set_ylim(filtered_values.min(), filtered_values.max())

        fig.canvas.draw_idle()

    # Connect sliders to update function
    slider_block.on_changed(update)
    slider_key_start.on_changed(update)
    slider_key_end.on_changed(update)
    slider_steps.on_changed(update)
    slider_rounds.on_changed(update)

    # Add reset button
    resetax = plt.axes([0.8, 0.9, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        slider_block.reset()
        slider_key_start.reset()
        slider_key_end.reset()
        slider_steps.reset()
        slider_rounds.reset()

    button.on_clicked(reset)

    plt.show()



# ------------------------ Interactive Visual Menu ------------------------

def handle_visualize(block, key, rounds):
    while True:
        print(f"\nVisualization Options for block={block}, key={key}, rounds={rounds}")
        print("  1. 3D Encryption Surface")
        print("  2. 2D Heatmap")
        print("  3. Interactive Bifurcation Diagram")
        print("  4. Cancel\n")

        choice = input("Choose an option (1-4): ").strip()
        if choice == "1":
            try:
                print("Launching 3D Encryption Surface visualization...")
                result = subprocess.run([
                    sys.executable, "Encryption_surface3D.py",
                    "plot",
                    "--block", str(block),
                    "--key-start", str(key),
                    "--key-end", str(key + 1),
                    "--steps", "50",
                    "--rounds", str(rounds),
                    "--color-by", "waveform",
                    "--interactive"
                ])
                print(f"3D visualization exited with code {result.returncode}")
            except Exception as e:
                print(f"Failed to launch 3D viewer: {e}")

        elif choice == "2":
            try:
                print("Launching 2D Heatmap visualization...")
                result = subprocess.run([
                    sys.executable, "Heatmap2D.py",
                    "--block", str(block),
                    "--key", str(key),
                    "--rounds", str(rounds)
                ])
                print(f"2D Heatmap exited with code {result.returncode}")
            except Exception as e:
                print(f"Failed to launch 2D Heatmap: {e}")

        elif choice == "3":
            print("Launching interactive Bifurcation Diagram visualization...")
            interactive_bifurcation()

        elif choice == "4":
            print("Returning to main menu...\n")
            break

        else:
            print("Invalid selection. Please choose 1, 2, 3, or 4.")



# Optional: simple test run if this file is executed directly
if __name__ == "__main__":
    block = 0.5
    key = 0.1
    rounds = 100
    handle_visualize(block, key, rounds)
