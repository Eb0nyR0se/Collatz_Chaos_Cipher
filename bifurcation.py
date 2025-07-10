# File: bifurcation.py

import matplotlib.pyplot as plt
import numpy as np
from main import signal_spiral_encrypt

def bifurcation_diagram(block, key_start, key_end, steps=200, rounds=30):
    keys = np.linspace(key_start, key_end, steps)  # float keys
    last_values = []

    for k in keys:
        _, history, _ = signal_spiral_encrypt(block, k, rounds=rounds)  # k as float
        last_values.append(history[-1][0])  # last block value

    plt.figure(figsize=(10, 6))
    plt.plot(keys, last_values, ',k', alpha=0.5)
    plt.title("Bifurcation Diagram (Final value vs. Key)")
    plt.xlabel("Key")
    plt.ylabel("Final Value After Encryption")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    bifurcation_diagram(0x112233, 1_000_000.0, 10_000_000.0, steps=1000, rounds=50)
