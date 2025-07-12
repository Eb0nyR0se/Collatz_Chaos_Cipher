# File: phase_space.py

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
from cipher_core import signal_spiral_encrypt


def interactive_phase_space():
    """
    Interactive phase space plot with sliders for block, key, and number of rounds.
    """
    # Default parameters
    block = 0.5
    key = 0.5
    rounds = 100

    # Initial encryption to populate plot
    ciphertext, history, _ = signal_spiral_encrypt(block, key, rounds=rounds)
    values = [h[0] for h in history]
    x = values[:-1]
    y = values[1:]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.1, bottom=0.3)
    scatter = ax.scatter(x, y, s=10, alpha=0.6, c='purple')
    ax.set_title("Phase Space Plot")
    ax.set_xlabel("Value at step n")
    ax.set_ylabel("Value at step n+1")
    ax.grid(True)

    # Slider axes
    ax_block = plt.axes([0.1, 0.2, 0.8, 0.03])
    ax_key = plt.axes([0.1, 0.15, 0.8, 0.03])
    ax_rounds = plt.axes([0.1, 0.1, 0.8, 0.03])

    # Create sliders
    slider_block = Slider(ax_block, 'Block', 0.0, 1.0, valinit=block, valstep=0.01)
    slider_key = Slider(ax_key, 'Key', 0.0, 1.0, valinit=key, valstep=0.01)
    slider_rounds = Slider(ax_rounds, 'Rounds', 10, 500, valinit=rounds, valstep=1)

    def update(val):
        b = slider_block.val
        k = slider_key.val
        r = int(slider_rounds.val)
        _, history, _ = signal_spiral_encrypt(b, k, rounds=r)
        values = [h[0] for h in history]
        x = values[:-1]
        y = values[1:]

        # Update scatter data
        scatter.set_offsets(np.column_stack((x, y)))
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    slider_block.on_changed(update)
    slider_key.on_changed(update)
    slider_rounds.on_changed(update)

    # Reset button
    reset_ax = plt.axes([0.8, 0.925, 0.1, 0.04])
    button = Button(reset_ax, 'Reset', hovercolor='0.975')

    def reset(event):
        slider_block.reset()
        slider_key.reset()
        slider_rounds.reset()

    button.on_clicked(reset)

    plt.show()


if __name__ == "__main__":
    interactive_phase_space()
