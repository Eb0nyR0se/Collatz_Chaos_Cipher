import matplotlib.pyplot as plt
import numpy as np
from cipher_core 
import signal_spiral_encrypt

def bifurcation_diagram(block, key_start, key_end, steps=200, rounds=30):
    keys = np.linspace(key_start, key_end, steps, dtype=np.uint64)
    last_values = []

    for k in keys:
        _, history = signal_spiral_encrypt(block, int(k), rounds=rounds)
        last_values.append(history[-1][0])  # last block value

    plt.figure(figsize=(10, 6))
    plt.plot(keys, last_values, ',k', alpha=0.5)
    plt.title("Bifurcation Diagram (Final value vs. Key)")
    plt.xlabel("Key")
    plt.ylabel("Final Value After Encryption")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    bifurcation_diagram(0x112233, 1_000_000, 10_000_000, steps=1000, rounds=50)
