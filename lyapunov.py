# File: lyapunov.py

import numpy as np
from cipher import signal_spiral_encrypt

def lyapunov_exponent(block, key, rounds=100, delta=1e-9):
    _, history1, _ = signal_spiral_encrypt(block, key, rounds=rounds)
    _, history2, _ = signal_spiral_encrypt(block + delta, key, rounds=rounds)

    distances = []
    for (b1, _, _), (b2, _, _) in zip(history1, history2):
        distances.append(abs(b2 - b1))

    distances = np.array(distances)
    distances[distances == 0] = 1e-10  # avoid log(0)

    lyapunov = np.mean(np.log(distances / delta))
    print(f"Approximate Lyapunov exponent: {lyapunov:.5f}")

if __name__ == "__main__":
    lyapunov_exponent(0x112233, 0x4242424242424242, rounds=200)
