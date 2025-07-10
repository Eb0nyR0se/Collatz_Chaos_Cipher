import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
from cipher_core import signal_spiral_encrypt

def surface_plot(block, key_start, key_end, steps=100, rounds=50):
    keys = np.linspace(key_start, key_end, steps, dtype=np.uint64)
    values_matrix = np.zeros((steps, rounds))

    for i, k in enumerate(keys):
        _, history = signal_spiral_encrypt(block, int(k), rounds=rounds)
        values_matrix[i, :] = [h[0] for h in history]

    X, Y = np.meshgrid(np.arange(rounds), np.arange(steps))
    Z = values_matrix

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Round')
    ax.set_ylabel('Key Index')
    ax.set_zlabel('Block Value')
    ax.set_title('3D Surface Plot of Block Values Over Rounds and Keys')
    plt.show()

if __name__ == "__main__":
    surface_plot(0x123456, 100_000, 1_000_000, steps=200, rounds=100)
