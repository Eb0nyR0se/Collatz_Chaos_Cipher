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
    block = prompt_float("Enter block (as float)", 0x112233)
    key_start = prompt_float("Enter start key range", 1_000_000.0)
    key_end = prompt_float("Enter end key range", 10_000_000.0)
    steps = prompt_int("Enter number of steps", 1000)
    rounds = prompt_int("Enter number of rounds", 50)

    if key_start >= key_end:
        print("Error: key_start must be less than key_end.")
    else:
        bifurcation_diagram(block, key_start, key_end, steps, rounds)
