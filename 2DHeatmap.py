import matplotlib.pyplot as plt
import numpy as np
from cipher_core import signal_spiral_encrypt

def heatmap_multiple_runs(block, key_start, key_end, steps=100, rounds=50):
    keys = np.linspace(key_start, key_end, steps, dtype=np.uint64)
    values_matrix = np.zeros((steps, rounds))

    for i, k in enumerate(keys):
        _, history = signal_spiral_encrypt(block, int(k), rounds=rounds)
        values_matrix[i, :] = [h[0] for h in history]

    plt.figure(figsize=(12, 6))
    plt.imshow(values_matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label='Block Value')
    plt.title("Heatmap of Block Values Over Rounds and Keys")
    plt.xlabel("Round")
    plt.ylabel("Key Index")
    plt.show()

if __name__ == "__main__":
    heatmap_multiple_runs(0x123456, 100_000, 1_000_000, steps=200, rounds=100)
