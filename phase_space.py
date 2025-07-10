#File: phase_space.py

import matplotlib.pyplot as plt
from cipher import signal_spiral_encrypt

def phase_space_plot(block, key, rounds=100):
    ciphertext, history = signal_spiral_encrypt(block, key, rounds=rounds)
    values = [h[0] for h in history]

    x = values[:-1]
    y = values[1:]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=10, alpha=0.6)
    plt.title("Phase Space Plot")
    plt.xlabel("Value at step n")
    plt.ylabel("Value at step n+1")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    phase_space_plot(0x1122334455, 0x4242424242424242, rounds=200)

