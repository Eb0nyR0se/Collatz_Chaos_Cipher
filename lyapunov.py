# File: lyapunov.py

import numpy as np
from main import signal_spiral_encrypt

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

def prompt_float(prompt_text, default):
    while True:
        try:
            val = input(f"{prompt_text} [default: {default}]: ").strip()
            if val == "":
                return default
            return float(val)
        except ValueError:
            print("Invalid input. Please enter a valid float.")

def prompt_int(prompt_text, default):
    while True:
        try:
            val = input(f"{prompt_text} [default: {default}]: ").strip()
            if val == "":
                return default
            iv = int(val)
            if iv <= 0:
                print("Please enter a positive integer.")
                continue
            return iv
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

if __name__ == "__main__":
    block = prompt_float("Enter block (float)", 0x112233)
    key = prompt_float("Enter key (float)", 0x4242424242424242)
    rounds = prompt_int("Enter number of rounds", 200)
    delta = prompt_float("Enter delta (small float perturbation)", 1e-9)

    lyapunov_exponent(block, key, rounds, delta)
