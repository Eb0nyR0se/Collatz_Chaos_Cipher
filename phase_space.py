# File: phase_space.py

import matplotlib.pyplot as plt
from main import signal_spiral_encrypt


def phase_space_plot(init_block, init_key, num_rounds=100):
    """
    Generate a phase space plot from the signal_spiral_encrypt history.
    """
    ciphertext, history, _ = signal_spiral_encrypt(init_block, init_key, rounds=num_rounds)
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
                return float(default)
            if val.lower().startswith("0x"):
                return float(int(val, 16))
            return float(val)
        except ValueError:
            print("Invalid input. Please enter a float or hex (e.g., 0x1234).")


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
    user_block = prompt_float("Enter block (float or hex)", "0x1122334455")
    user_key = prompt_float("Enter key (float or hex)", "0x4242424242424242")
    user_rounds = prompt_int("Enter number of rounds", 200)

    phase_space_plot(user_block, user_key, user_rounds)
