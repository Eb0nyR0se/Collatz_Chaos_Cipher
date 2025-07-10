# File: phase_space.py

import matplotlib.pyplot as plt
from main import signal_spiral_encrypt

def phase_space_plot(block, key, rounds=100):
    ciphertext, history, _ = signal_spiral_encrypt(block, key, rounds=rounds)
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
    block = prompt_float("Enter block (float)", 0x1122334455)
    key = prompt_float("Enter key (float)", 0x4242424242424242)
    rounds = prompt_int("Enter number of rounds", 200)

    phase_space_plot(block, key, rounds)
